from CCKRM_Data import CCKRM_Data
from util import *
from sparsemax import Sparsemax

class CCKRM(nn.Module):
    def __init__(self, args, student_size, prior_course_size, concur_course_size,
                 target_course_size, n_prior, n_concur,
                 pretrained_p_file=None, pretrained_r_file=None,
                 pretrained_x_file=None):

        super(CCKRM, self).__init__()
        self.args               = args
        self.embedding_size     = args.embedding_size
        self.lrn_rate           = args.lrn_rate
        self.l2_reg             = args.l2_reg
        self.train_loss         = args.train_loss
        self.verbose            = args.verbose
        self.attn_weight_size   = args.attn_weight_size
        self.grade_b4_attn      = args.grade_b4_attn
        self.prior_beta         = args.prior_beta
        self.row_center_grades  = args.row_center_grades
        self.compute_sparsemax  = args.sparsemax
        self.temp               = args.temp
        self.accumulate         = args.accumulate
        self.pretrained_p_file  = pretrained_p_file
        self.pretrained_r_file  = pretrained_r_file
        self.pretrained_x_file  = pretrained_x_file

        self.student_size       = student_size
        self.prior_course_size  = prior_course_size
        self.concur_course_size = concur_course_size
        self.target_course_size = target_course_size
        self.n_priors           = n_prior
        self.n_concur           = n_concur

        if self.compute_sparsemax == 1:
            self.sparsemax = Sparsemax(dim=1, temp=self.temp)

        self.pretrain = False
        if self.pretrained_p_file is not None:
            assert self.pretrained_r_file is not None and \
                self.pretrained_x_file is not None, "All of P and R, and X "
            "pretrained vectors should be given!"
            self.pretrain = True

        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.cuda()
            logging.info("CUDA is available")

        self._initialize_model()
        self._create_optimizer()
        self._create_criterion()

    def _create_optimizer(self):
        # self.optimizer = optim.SGD(
        #     self.model.parameters(), lr=self.lrn_rate,
        #     weight_decay=self.l2_reg) #, momentum=0.9)
        self.optimizer = optim.Adam(self.parameters(), lr=self.lrn_rate,
                                    weight_decay=self.l2_reg)

        return

    def _create_criterion(self):
        self.criterion = MSELoss()

        return

    def _initialize_model(self):
        if not self.row_center_grades:
            self.bias_S_      = nn.Embedding(self.student_size, 1) # student bias
        self.embedding_P_ = nn.Embedding(self.prior_course_size, self.embedding_size,
                                         padding_idx=0) # prior course embeddings
        self.bias_R_      = nn.Embedding(self.prior_course_size, 1) # target course bias
        self.embedding_R_ = nn.Embedding(self.target_course_size, self.embedding_size) # target course embeddings

        self.embedding_X_ = nn.Embedding(self.concur_course_size, self.embedding_size,
                                         padding_idx=0) # concurrent course embeddings

        if not self.row_center_grades:
            self._init_weights(self.bias_S_)
        self._init_weights(self.embedding_P_)

        if self.pretrain:
            self._load_pretrained_vectors(self.embedding_P_, self.pretrained_p_file)
            self._load_pretrained_vectors(self.embedding_R_, self.pretrained_r_file)
            self._load_pretrained_vectors(self.embedding_X_, self.pretrained_x_file)

        else:
            self._init_weights(self.embedding_P_)
            self._init_weights(self.embedding_R_)
            self._init_weights(self.embedding_X_)

        self._init_weights(self.bias_R_)

        # attention weights for prior courses
        self.attn_W_prior = nn.Linear(self.embedding_size, self.attn_weight_size,
                                      bias=True)
        self.attn_h_prior    = nn.Linear(self.attn_weight_size, 1, bias=False)

        # attention weights for concurrent courses
        self.attn_W_concur = nn.Linear(self.embedding_size, self.attn_weight_size,
                                      bias=True)
        self.attn_h_concur    = nn.Linear(self.attn_weight_size, 1, bias=False)
        return

    def _load_pretrained_vectors(self, embedding, infile):

        embed_nrows, embed_ncols = int(embedding.weight.data.shape[0]), int(embedding.weight.data.shape[1])
        pretrained = []
        for line in readlines(infile):
            vec = map(float, line.split(" "))
            pretrained.append(vec)

        nrows, ncols = len(pretrained), len(pretrained[0])

        assert nrows == embed_nrows, "File {} has {} rows while embedding "
        "has {}!".format(nrows, embed_nrows)

        assert ncols == embed_ncols, "File {} has {} ncols while embedding "
        "has {}!".format(ncols, embed_ncols)

        embedding.weight.data = torch.FloatTensor(pretrained)

        return

    def _init_weights(self, embedding, size=1):

        # init_range = 1/math.sqrt(size)
        # init_range = 0.001
        embedding.weight.data.normal_(mean=0.0, std=0.01)*size

        return

    def forward(self, sid, prior_cids, prior_grades, concur_cids, target_cid):
        """
        Returns:
            Loss of this process, a pytorch variable.
        """
        if not self.row_center_grades:
            self.bias_s_      = self.bias_S_(sid).squeeze(dim=1)
        self.embedding_p_ = self.embedding_P_(prior_cids)
        self.embedding_x_ = self.embedding_P_(concur_cids)
        # self.embedding_x_ = self.embedding_X_(concur_cids)
        self.prior_grades = prior_grades.view(-1, self.n_priors, 1)

        self.embedding_r_ = self.embedding_R_(target_cid)
        concur_attn_weights = self._compute_contextual_target_embedding(concur_cids)

        self.ks_, prior_attn_weights = self._compute_knowledge_state()

        self.bias_r_      = torch.sum(self.bias_R_(prior_cids), 1).squeeze(dim=1)
        if not self.row_center_grades:
            biases = self.bias_s_ + self.bias_r_
        else:
            biases = self.bias_r_

        ks_r = self.ks_*self.embedding_r_.squeeze(dim=1)

        out = torch.sum(ks_r, 1)
        out = biases + out

        return out, prior_attn_weights, concur_attn_weights

    def _compute_knowledge_state(self):

        A = None

        if self.accumulate == 0: # attention mechanism
            if self.grade_b4_attn:
                query = self.prior_grades * self.embedding_p_
            else:
                query = self.embedding_p_

            mask = self.prior_grades.ne(0.0).float().view(-1, self.n_priors)

            A = self._attention_MLP(key=self.embedding_r_,
                                    query=query,
                                    query_size=self.embedding_size,
                                    mask=mask,
                                    W=self.attn_W_prior,
                                    h=self.attn_h_prior,
                                    beta=self.prior_beta)

            self.embedding_p_ = self.prior_grades * self.embedding_p_
            ks = torch.sum(A.unsqueeze(2) * self.embedding_p_, 1)

        elif self.accumulate == 1: # max of prior courses
            mask = self.prior_grades.ne(0.0).float()
            mask = mask.view(-1, self.n_priors, 1)*self.embedding_p_
            mask = mask.ne(0.0).float()
            one_minus_mask = 1.0-mask
            max_mask = -99999999.0*one_minus_mask
            self.embedding_p_ = self.prior_grades * self.embedding_p_
            embedding_p = self.embedding_p_ + max_mask
            ks, _ = embedding_p.max(dim=1)

        return ks, A

    def _compute_contextual_target_embedding(self, concur_cids):

        attn_weights = None
        mask = concur_cids.ne(0).float() # assuming a padding idx of 0

        if self.accumulate == 0:
            attn_weights = self._attention_MLP(key=self.embedding_r_,
                                               query=self.embedding_x_,
                                               query_size=self.embedding_size,
                                               mask=mask,
                                               W=self.attn_W_concur,
                                               h=self.attn_h_concur)

            embedding_x = torch.sum(attn_weights.unsqueeze(2)*self.embedding_x_, 1)

        elif self.accumulate == 1:
            mask = mask.view(-1, self.n_concur, 1)*self.embedding_x_
            mask = mask.ne(0.0).float()
            one_minus_mask = 1.0-mask
            max_mask = -99999999.0*one_minus_mask
            embedding_x = self.embedding_x_ + max_mask
            embedding_x, _ = embedding_x.max(dim=1)

        self.embedding_r_ = torch.mul(embedding_x, self.embedding_r_.squeeze(dim=1))
        self.embedding_r_ = self.embedding_r_.unsqueeze(dim=1)

        return attn_weights

    def _attention_MLP(self, query=None, query_size=None,
                       key=None, mask=None,
                       W=None, h=None,
                       beta=1.0):

        qk = query*key
        b, n = qk.shape[0], qk.shape[1]
        MLP_output = W(qk) # (b, n, attn_size)
        MLP_output = F.relu(MLP_output)

        A_ = h(MLP_output).view(b, n)

        if self.compute_sparsemax == 0: # softmax
            A = softmax(A_, mask=mask, beta=beta)

        else: # sparsemax
            A = self.sparsemax(A_, mask=mask)

        return A

def load_data(dataset, batch_size):
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=True)
                      # num_workers=8)

def training(model, dataset, dataloader, epochs):
    """Multiple training.

    Returns:
        None.
    """
    data_size = len(dataset)

    best_train_mae, best_val_mae = 100.0, 100.0
    best_train_rmse, best_val_rmse = 100.0, 100.0
    tr_mae, train_rmse = 0.0, 0.0
    best_model = None
    bad_epochs = 10
    logging.info("train-count={} bad-epochs={}\n".format(data_size,
                                                         bad_epochs))
    cur_bad_epochs = 0
    lr = model.lrn_rate

    train_maes, val_maes = list(), list()

    torch.manual_seed(int(time.time()))

    start = time.time()
    start_time = time.time()

    # evaluate model with random initialization
    # if model.train_loss:
    #     tr_mae, train_rmse  = evaluate_loss(model, dataset, dataset.train, train=True)

    # val_mae, val_rmse   = evaluate_loss(model, dataset, dataset.val)
    # logging.info("[Before training] "
    #              "train-MAE:{:.5f}, "
    #              "val-MAE:{:.5f}".format(tr_mae,
    #                                     val_mae))

    for epoch in xrange(epochs):

        if cur_bad_epochs == bad_epochs:
            break

        model.train(True)

        total_tr_loss = 0.0

        for i, batch in enumerate(dataloader):
            total_tr_loss += batch_train(batch, model)

        if epoch % model.verbose == 0:
            model.train(False) # useful when using dropout regularization
            tr_mae, train_rmse = 0.0, 0.0
            if model.train_loss:
	        tr_mae, train_rmse  = evaluate_loss(model, dataset, dataset.train, train=True)

	    val_mae, val_rmse   = evaluate_loss(model, dataset, dataset.val)

            train_maes.append(tr_mae)
            val_maes.append(val_mae)

	    if val_mae < best_val_mae:
                best_train_mae  = tr_mae
                best_val_mae    = val_mae
                best_train_rmse = train_rmse
                best_val_rmse   = val_rmse

                new_model = CCKRM(model.args,
                                  model.student_size,
                                  model.prior_course_size,
                                  model.concur_course_size,
                                  model.target_course_size,
                                  model.n_priors,
                                  model.n_concur)
		copy_model(new_model, model)
		best_model = new_model
                cur_bad_epochs = 0
            else:
                cur_bad_epochs += 1

            epoch_time = timeSince(start_time)
            start_time = time.time()
            logging.info("[{}:{:}] "
                         "train-MAE:{:.5f}, "
                         "val-MAE:{:.5f}, "
                         "best: "
                         "MAE:{:.5f} RMSE:{:.5f}, "
                         "bad-epochs={}".format(str(epoch+1).zfill(4),
                                                str(epoch_time).zfill(8),
                                                tr_mae,
                                                val_mae,
                                                best_val_mae,
                                                best_val_rmse,
                                                cur_bad_epochs))

            if cur_bad_epochs == bad_epochs:
                break

    model.train(False)

    logging.info("Time taken: {}".format(timeSince(start)))

    # val_gpa_maes = [self.val_gpa_mae]*len(train_maes)
    # plot([train_maes, val_maes, val_gpa_maes],
    #      outfile="{}_train_val_maes.png".format(self.outfile_pref),
    #      legend=["train MAE", "validation MAE", "validation GPA-based MAE"])

    return(best_train_mae, best_val_mae, best_model)

def batch_train(batch, model):

    sid          = Variable(batch['sid'])
    prior_cids   = Variable(batch['prior_cids'])
    prior_grades = Variable(batch['prior_grades'])
    concur_cids  = Variable(batch['concur_cids'])
    target_cid   = Variable(batch['target_cid'])
    target_grade = Variable(batch['target_grade'])

    if model.use_cuda:
        sid          = sid.cuda()
        prior_cids   = prior_cids.cuda()
        prior_grades = prior_grades.cuda()
        concur_cids  = concur_cids.cuda()
        target_cid   = target_cid.cuda()
        target_grade = target_grade.cuda()

    predictions, prior_attn_weights, concur_attn_weights = model.forward(sid,
                                                                         prior_cids,
                                                                         prior_grades,
                                                                         concur_cids,
                                                                         target_cid)
    loss = model.criterion(predictions, target_grade)
    tr_loss = loss.data[0]
    model.zero_grad()
    loss.backward()
    # self.optimizer.step(closure)
    model.optimizer.step()

    return tr_loss

def evaluate_loss(model, dataset, data, train=False,
                  save_predictions=False, outfile_pref=None):

    if save_predictions:
        assert outfile_pref is not None, "Inside evaluate_loss: save_predictions is "
        "set to True but no output file prefix is given!"
        fout_pred = open(outfile_pref+".test.predictions", 'w')
        fout_prior_attn = open(outfile_pref+".test.prior_attn_weights", 'w')
        fout_concur_attn = open(outfile_pref+".test.concur_attn_weights", 'w')

    actual, final_pred, initial_pred = list(), list(), list()

    students = dict()

    for i in xrange(len(data)):
        if train:
            example = dataset[i]
        else:
            example = dataset.prepare_sample(data[i])

        sid          = Variable(example['sid'].unsqueeze(0))
        prior_cids   = Variable(example['prior_cids'].unsqueeze(0))
        prior_grades = Variable(example['prior_grades'].unsqueeze(0))
        concur_cids   = Variable(example['concur_cids'].unsqueeze(0))
        target_cid   = Variable(example['target_cid'].unsqueeze(0))
        target_grade = example['target_grade']
        avg_prev_grade = example['avg_prev_grade']

        if model.use_cuda:
            sid          = sid.cuda()
            prior_cids   = prior_cids.cuda()
            prior_grades = prior_grades.cuda()
            concur_cids  = concur_cids.cuda()
            target_cid   = target_cid.cuda()
            target_grade = target_grade.cuda()

        prediction, prior_attn_weights, concur_attn_weights = model.forward(sid,
                                                                            prior_cids,
                                                                            prior_grades,
                                                                            concur_cids,
                                                                            target_cid)

        actual_grade = target_grade[0] + avg_prev_grade[0]
        pred_grade = prediction.data.squeeze()[0] + avg_prev_grade[0]

        if save_predictions:
            # prior attn weights
            if prior_attn_weights is not None:
                cid_weight = []
                for i in xrange(len(example['prior_cids'])):
                    if example['prior_grades'][i] != 0.0:
                        cid_weight.append(example['prior_cids'][i])
                        cid_weight.append("{:.3f}".format(example['prior_grades'][i]))
                        cid_weight.append(prior_attn_weights[0][i].data[0])

                fout_prior_attn.write("{} {} {:.3f} {:.3f} {}\n".format(example['sid'][0],
                                                                        example['target_cid'][0],
                                                                        actual_grade,
                                                                        pred_grade,
                                                                        " ".join(map(str, cid_weight))))

            # concurrent attn weights
            if concur_attn_weights is not None: # there exists at least 1 concurrent course in this example
                cid_weight = []
                for i in xrange(len(example['concur_cids'])):
                    if example['concur_cids'][i] == 0:
                        break
                    cid_weight.append(example['concur_cids'][i])
                    cid_weight.append(concur_attn_weights[0][i].data[0])

                fout_concur_attn.write("{} {} {:.3f} {:.3f} {}\n".format(example['sid'][0],
                                                                         example['target_cid'][0],
                                                                         actual_grade,
                                                                         pred_grade,
                                                                         " ".join(map(str, cid_weight))))

        actual.append(actual_grade)
        final_pred.append(pred_grade)
        if save_predictions:
            fout_pred.write("{} {} {:.3f} {:.3f}\n".format(example['sid'][0],
                                                      example['target_cid'][0],
                                                      actual_grade,
                                                      pred_grade))

    if save_predictions:
        fout_pred.close()
        fout_prior_attn.close()
        fout_concur_attn.close()
    # print actual
    # print final_pred
    return(mean_absolute_error(actual, final_pred),
           math.sqrt(mean_squared_error(actual, final_pred)))

def compute_test_data(indata):
    """
    Remove one concurrent course at a time to compute the prediction without it
    """
    data = []

    for i in xrange(len(indata)):
        example = indata[i]
        for j in xrange(len(indata[i]['concur_cids'])):
            new_concur_cids = [cid for cid in indata[i]['concur_cids']]
            new_concur_cids[j] = 0

            datum = {'sid': indata[i]['sid'],
                     'prior_cids': indata[i]['prior_cids'],
                     'prior_grades': indata[i]['prior_grades'],
                     'concur_cids': new_concur_cids,
                     'target_cid': indata[i]['target_cid'],
                     'target_grade': indata[i]['target_grade'],
                     'avg_prev_grade': indata[i]['avg_prev_grade']}

            data.append(datum)

    return data

def parse_args():
    argparser = argparse.ArgumentParser(description="Runs CCKRM")

    argparser.add_argument("infile_pref", help="Path prefix for input files ", type=str)
    argparser.add_argument("outdir", help="Path for output files", type=str)
    argparser.add_argument("--nprior", default=4, type=int, help="Min # prior courses for predicting a target course's grade. Default=4")
    argparser.add_argument("--min_est_count", default=10, type=int,
                           help="Min frequency of a course in the training set to be considered in the validation or test set. Default=10")
    argparser.add_argument("--row_center_grades", choices=[0, 1], default=0, type=int,
                           help="Whether to row center student's grades. Default=0")
    argparser.add_argument("--grade_b4_attn", default=0, choices=[0, 1], type=int,
                           help="Whether to weigh prior courses with their grades"
                           "before computing their attention weights (1) or not (0)."
                           " Default=1")
    argparser.add_argument("--accumulate", default=0, choices=[0, 1], type=int,
                           help="(0) for attention mechanism, or "
                           " (1) for max pooling. Default=0")
    argparser.add_argument("--prior_beta", type=float, default=1.0,
                           help="Index of coefficient of sum of exp(A) for attention "
                           "weights of prior courses."
                           " Default=1.0")
    argparser.add_argument("--embedding_size", help="Default=10", type=int, default=10)
    argparser.add_argument("--l2_reg", help="Default=1e-7", type=float, default=1e-7)
    argparser.add_argument("--lrn_rate", help="Default=0.01", type=float, default=0.01)
    argparser.add_argument("--apply_decay", default=0, type=int, choices=[0, 1],
                           help="Apply decay on prior courses wrt time or not. Default=0")
    argparser.add_argument("--lamda", default=0, type=float,
                           help="Decay constant on prior grades (if apply_decay=1). Default=0")
    argparser.add_argument("--attn_weight_size", default=1, type=int,
                           help="Embedding size for attention weights. Default=1")
    argparser.add_argument("--sparsemax", choices=[0, 1], default=0, type=int,
                           help="Whether to perform softmax (0) or sparsemax (1) "
                           "on the attention weights. Default=0")
    argparser.add_argument("--temp", default=1.0, type=float,
                           help="Temperature parameter for sparsemax activation function. Default=1.0")
    argparser.add_argument("--epochs", default=100, type=int,
                           help="Number of training iterations. Default=100")
    argparser.add_argument("--batch_size", default=200, type=int,
                           help="Number of samples to consider in one batch. Default=200")
    argparser.add_argument("--train_loss", default=1, type=int, choices=[0, 1],
                           help="Calculate training loss or not. Default=1")
    argparser.add_argument("--pretrained_p_file", type=str, help="File containing"
                           " pretrained vectors for the prior course embeddings")
    argparser.add_argument("--pretrained_r_file", type=str, help="File containing"
                           " pretrained vectors for the target course embeddings")
    argparser.add_argument("--pretrained_x_file", type=str, help="File containing"
                           " pretrained vectors for the concurrent course embeddings")
    argparser.add_argument('--verbose', type=int, default=5,
                        help='Interval of evaluation. Default=5.')

    return argparser.parse_args()

if __name__ == '__main__':

    args = parse_args()

    outfile_pref = "{}/dim{}_lr{}_l2{}".format(args.outdir,
                                               args.embedding_size,
                                               args.lrn_rate,
                                               args.l2_reg)
    if args.accumulate == 0:
        outfile_pref = "{}_attndim{}".format(outfile_pref,
                                             args.attn_weight_size)

    outfile_pref = "{}_beta{}".format(outfile_pref, args.prior_beta)

    if args.accumulate == 1:
        outfile_pref = "{}_decay{}_lamda{}".format(outfile_pref,
                                                   args.apply_decay,
                                                   args.lamda)

    if args.accumulate == 0 and args.sparsemax == 1:
        outfile_pref = "{}_temp{}".format(outfile_pref,
                                          args.temp)


    log_file = outfile_pref + ".log"
    logging.basicConfig(filename=log_file,
                        filemode='w',
                        level=logging.INFO)

    logging.info("Starting training CCKRM model ......")
    logging.info(args)

    dataset = CCKRM_Data(args)
    logging.info("#students={}, #courses={}".format(dataset.num_students,
                                                    dataset.num_courses))
    logging.info("size(train)={}".format(len(dataset.train)))
    logging.info("size(val)={}".format(len(dataset.val)))
    logging.info("size(test)={}".format(len(dataset.test)))
    logging.info("mean(npriors)={:.2f}, variance(npriors)={:.2f}".format(np.mean(dataset.npriors),
                                                                        np.std(dataset.npriors)))

    dataloader = load_data(dataset, args.batch_size)

    model = CCKRM(args,
                  dataset.max_student_size,
                  dataset.max_prior_size,
                  dataset.max_concur_size,
                  dataset.max_target_size,
                  dataset.max_n_prior,
                  dataset.max_n_concur)

    train_mae, val_mae, best_model = training(model, dataset, dataloader,
                                              args.epochs)

    # test_data = compute_test_data(dataset.test)

    test_mae, test_rmse = evaluate_loss(best_model, dataset, dataset.test,
                                        save_predictions=True,
                                        outfile_pref=outfile_pref)

    logging.info("best-train-mae {:.5f}".format(train_mae))
    logging.info("best-val-mae {:.5f} "
                 "test-mae {:.5f} "
                 "test-rmse {:.5f}".format(val_mae, test_mae, test_rmse))
