from CCKRM_Data import CCKRM_Data
from util import *
from sparsemax import Sparsemax

class CKRM(nn.Module):
    def __init__(self, args, student_size, prior_course_size,
                 target_course_size, n_prior,
                 pretrained_p_file=None, pretrained_r_file=None):
        super(CKRM, self).__init__()
        self.args               = args
        self.embedding_size     = args.embedding_size
        self.lrn_rate           = args.lrn_rate
        self.l2_reg             = args.l2_reg
        self.train_loss         = args.train_loss
        self.verbose            = args.verbose
        self.accumulate         = args.accumulate_priors
        self.attn_weight_size   = args.attn_weight_size
        self.grade_b4_attn      = args.grade_b4_attn
        self.beta               = args.beta
        self.row_center_grades  = args.row_center_grades
        self.compute_sparsemax  = args.sparsemax
        self.temp                = args.temp
        self.pretrained_p_file  = pretrained_p_file
        self.pretrained_r_file  = pretrained_r_file

        self.student_size       = student_size
        self.prior_course_size  = prior_course_size
        self.target_course_size = target_course_size
        self.n_priors           = n_prior

        if self.compute_sparsemax == 1:
            self.sparsemax = Sparsemax(dim=1, temp=self.temp)

        self.pretrain = False
        if self.pretrained_p_file is not None:
            assert self.pretrained_r_file is not None, "Both the P and R "
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
        self.optimizer = optim.Adam(self.parameters(), lr=self.lrn_rate,
                                    weight_decay=self.l2_reg)

        return

    def _create_criterion(self):
        self.criterion = MSELoss()

        return

    def _initialize_model(self):
        if not self.row_center_grades:
            self.bias_S_      = nn.Embedding(self.student_size, 1) # student bias

        self.bias_R_      = nn.Embedding(self.prior_course_size, 1) # target course bias

        self.embedding_P_ = nn.Embedding(self.prior_course_size, self.embedding_size,
                                         padding_idx=0) # prior course embeddings
        self.embedding_R_ = nn.Embedding(self.target_course_size, self.embedding_size) # target course embeddings

        if not self.row_center_grades:
            self._init_weights(self.bias_S_)
        self._init_weights(self.bias_R_)

        if self.pretrain:
            self._load_pretrained_vectors(self.embedding_P_, self.pretrained_p_file)
            self._load_pretrained_vectors(self.embedding_R_, self.pretrained_r_file)

        else:
            self._init_weights(self.embedding_P_)
            self._init_weights(self.embedding_R_)

        if self.accumulate == 2: # attention weights
            self.attn_W = nn.Linear(self.embedding_size, self.attn_weight_size,
                                    bias=True)
            self.attn_h    = nn.Linear(self.attn_weight_size, 1, bias=False)

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

        embedding.weight.data.normal_(mean=0.0, std=0.01)*size

        return

    def forward(self, sid, prior_cids, prior_grades, target_cid):
        """
        Returns:
            Loss of this process, a pytorch variable.
        """
        if not self.row_center_grades:
            self.bias_s_      = self.bias_S_(sid).squeeze(dim=1)
        self.embedding_p_ = self.embedding_P_(prior_cids)
        self.prior_grades = prior_grades.view(-1, self.n_priors, 1)
        self.embedding_r_ = self.embedding_R_(target_cid)
        self.ks_, attn_weights = self._compute_knowledge_state()
        self.bias_r_      = torch.sum(self.bias_R_(prior_cids), 1).squeeze(dim=1)
        if not self.row_center_grades:
            biases = self.bias_s_ + self.bias_r_
        else:
            biases = self.bias_r_

        self.embedding_r_ = self.embedding_r_.squeeze(dim=1)
        ks_r = self.ks_*self.embedding_r_
        out = torch.sum(ks_r, 1)
        out = biases + out

        return out, attn_weights

    def _compute_knowledge_state(self):

        attn_weights = None

        if self.accumulate == 0: # sum of prior courses
            self.embedding_p_ = self.prior_grades * self.embedding_p_
            ks = torch.sum(self.embedding_p_, 1)

        elif self.accumulate == 1: # average of prior courses
            self.embedding_p_ = self.prior_grades * self.embedding_p_
            mask = self.prior_grades.ne(0.0).float()
            ks = mask.view(-1, self.n_priors, 1) * self.embedding_p_
            ks = torch.sum(ks, 1)
            count = torch.sum(mask, 1)
            ks = torch.div(ks, count)

        elif self.accumulate == 2: # attention weights
            if self.grade_b4_attn:
                query = self.prior_grades * self.embedding_p_
            else:
                query = self.embedding_p_

            mask = self.prior_grades.ne(0.0).float().view(-1, self.n_priors)
            attn_weights = self._attention_MLP(key=self.embedding_r_,
                                               query=query,
                                               query_size=self.embedding_size,
                                               mask=mask,
                                               W=self.attn_W,
                                               h=self.attn_h)

            self.embedding_p_ = self.prior_grades * self.embedding_p_
            ks = torch.sum(attn_weights.unsqueeze(2) * self.embedding_p_, 1)

        return ks, attn_weights

    def _attention_MLP(self, query=None, query_size=None, key=None, mask=None,
                       W=None, h=None):

        qk = query*key
        b, n = qk.shape[0], qk.shape[1]

        MLP_output = W(qk) # (b, n, attn_size)
        MLP_output = F.relu(MLP_output)

        A_ = h(MLP_output).view(b, n)

        if self.compute_sparsemax == 0: # softmax
            A = softmax(A_, mask=mask, beta=self.beta)

        else: # sparsemax
            A = self.sparsemax(A_, mask=mask)

        return A

    def write_embeddings(self, outfile_pref):

        self._write(self.embedding_P_, outfile_pref+".p")
        self._write(self.embedding_R_, outfile_pref+".r")

        return

    def _write(self, embedding, outfile):

        nrows, ncols = embedding.weight.data.shape

        fout = open(outfile, 'w')

        for i in xrange(nrows):
            vec = []
            for j in xrange(ncols):
                vec.append(embedding.weight.data[i][j])
            fout.write(" ".join(map(str, vec))+"\n")

        fout.close()

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

                new_model = CKRM(model.args,
                                 model.student_size,
                                 model.prior_course_size,
                                 model.target_course_size,
                                 model.n_priors)
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

    return(best_train_mae, best_val_mae, best_model)

def batch_train(batch, model):

    sid          = Variable(batch['sid'])
    prior_cids   = Variable(batch['prior_cids'])
    prior_grades = Variable(batch['prior_grades'])
    target_cid   = Variable(batch['target_cid'])
    target_grade = Variable(batch['target_grade'])

    if model.use_cuda:
        sid          = sid.cuda()
        prior_cids   = prior_cids.cuda()
        prior_grades = prior_grades.cuda()
        target_cid   = target_cid.cuda()
        target_grade = target_grade.cuda()

    predictions, attn_weights = model.forward(sid, prior_cids, prior_grades, target_cid)
    loss = model.criterion(predictions, target_grade)
    tr_loss = loss.data[0]

    model.zero_grad()
    loss.backward()
    model.optimizer.step()

    return tr_loss

def evaluate_loss(model, dataset, data, train=False,
                  save_predictions=False, outfile_pref=None):

    if save_predictions:
        assert outfile_pref is not None, "Inside evaluate_loss: save_predictions is "
        "set to True but no output file prefix is given!"
        fout_pred = open(outfile_pref+".predictions", 'w')
        if model.accumulate == 2:
            fout_attn = open(outfile_pref+".attn_weights", 'w')

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
        target_cid   = Variable(example['target_cid'].unsqueeze(0))
        target_grade = example['target_grade']
        avg_prev_grade = example['avg_prev_grade']

        if model.use_cuda:
            sid          = sid.cuda()
            prior_cids   = prior_cids.cuda()
            prior_grades = prior_grades.cuda()
            target_cid   = target_cid.cuda()
            target_grade = target_grade.cuda()

        prediction, attn_weights = model.forward(sid, prior_cids, prior_grades, target_cid)

        actual_grade = target_grade[0] + avg_prev_grade[0]
        pred_grade = prediction.data.squeeze()[0] + avg_prev_grade[0]

        if save_predictions and model.accumulate == 2:
            cid_weight = []
            for i in xrange(len(example['prior_cids'])):
                if example['prior_grades'][i] != 0.0:
                    cid_weight.append(example['prior_cids'][i])
                    cid_weight.append("{:.3f}".format(example['prior_grades'][i]))
                    cid_weight.append(attn_weights[0][i].data[0])

            fout_attn.write("{} {} {:.3f} {:.3f} {}\n".format(example['sid'][0],
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
        if model.accumulate == 2:
            fout_attn.close()

    return(mean_absolute_error(actual, final_pred),
           math.sqrt(mean_squared_error(actual, final_pred)))

def parse_args():
    argparser = argparse.ArgumentParser(description="Runs CKRM")

    argparser.add_argument("infile_pref", help="Path prefix for input files ", type=str)
    argparser.add_argument("outdir", help="Path for output files", type=str)
    argparser.add_argument("--nprior", default=4, type=int, help="Min # prior courses for predicting a target course's grade. Default=4")
    argparser.add_argument("--min_est_count", default=10, type=int,
                           help="Min frequency of a course in the training set to be considered in the validation or test set. Default=10")
    argparser.add_argument("--row_center_grades", choices=[0, 1], default=0, type=int,
                           help="Whether to row center student's grades. Default=0")
    argparser.add_argument("--sparsemax", choices=[0, 1], default=0, type=int,
                           help="Whether to perform softmax (0) or sparsemax (1) "
                           "on the attention weights. Default=0")
    argparser.add_argument("--temp", default=1.0, type=float,
                           help="Temperature parameter for sparsemax activation function. Default=1.0")
    argparser.add_argument("--accumulate_priors", default=0, choices=[0, 1, 2], type=int,
                           help="(0) for summing, (1) for averaging, "
                           "(2) for attention mechanism. Default=0")
    argparser.add_argument("--grade_b4_attn", default=0, choices=[0, 1], type=int,
                           help="Whether to weigh prior courses with their grades"
                           "before computing their attention weights (1) or not (0)."
                           " Default=1")
    argparser.add_argument("--beta", type=float, default=1.0,
                           help="Index of coefficient of sum of exp(A) for attention weights."
                           " Default=1.0")
    argparser.add_argument("--apply_decay", default=0, type=int, choices=[0, 1],
                           help="Apply decay on prior courses wrt time or not. Default=0")
    argparser.add_argument("--lamda", default=0, type=float,
                           help="Decay constant on prior grades (if apply_decay=1). Default=0")
    argparser.add_argument("--normalize_grades", default=0, type=int, choices=[0, 1],
                           help="Normalize grades between 0.0 and 1.0 or not. Default=0")
    argparser.add_argument("--embedding_size", help="Default=10", type=int, default=10)
    argparser.add_argument("--l2_reg", help="Default=1e-7", type=float, default=1e-7)
    argparser.add_argument("--lrn_rate", help="Default=0.01", type=float, default=0.01)
    argparser.add_argument("--attn_weight_size", default=1, type=int,
                           help="Embedding size for attention weights. Default=1")
    argparser.add_argument("--epochs", default=100, type=int,
                           help="Number of training iterations. Default=100")
    argparser.add_argument("--batch_size", default=200, type=int,
                           help="Number of samples to consider in one batch. Default=200")
    argparser.add_argument("--train_loss", default=1, type=int, choices=[0, 1],
                           help="Calculate training loss or not. Default=1")
    argparser.add_argument("--write_embeddings", default=0, type=int, choices=[0, 1],
                           help="Write embeddings to output files. Default=0")
    argparser.add_argument("--pretrained_p_file", type=str, help="File containing"
                           " pretrained vectors for the prior course embeddings")
    argparser.add_argument("--pretrained_r_file", type=str, help="File containing"
                           " pretrained vectors for the target course embeddings")
    argparser.add_argument('--verbose', type=int, default=5,
                        help='Interval of evaluation. Default=5.')

    return argparser.parse_args()

if __name__ == '__main__':

    args = parse_args()

    outfile_pref = "{}/dim{}_lr{}_l2{}".format(args.outdir,
                                               args.embedding_size,
                                               args.lrn_rate,
                                               args.l2_reg)

    if args.accumulate_priors in [0, 1, 3]:
        outfile_pref = "{}_decay{}_lamda{}".format(outfile_pref,
                                                   args.apply_decay,
                                                   args.lamda)

    if args.accumulate_priors == 2:
        outfile_pref = "{}_attndim{}_gradeb4attn{}_beta{}".format(outfile_pref,
                                                                  args.attn_weight_size,
                                                                  args.grade_b4_attn,
                                                                  args.beta)
        if args.sparsemax == 1:
            outfile_pref = "{}_temp{}".format(outfile_pref,
                                              args.temp)

    log_file = outfile_pref + ".log"
    logging.basicConfig(filename=log_file,
                        filemode='w',
                        level=logging.INFO)

    logging.info("Starting training CKRM model ......")
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

    model = CKRM(args,
                 dataset.max_student_size,
                 dataset.max_prior_size,
                 dataset.max_target_size,
                 dataset.max_n_prior,
                 pretrained_p_file=args.pretrained_p_file,
                 pretrained_r_file=args.pretrained_r_file)

    train_mae, val_mae, best_model = training(model, dataset, dataloader,
                                              args.epochs)

    _, _ = evaluate_loss(best_model, dataset, dataset.val,
                         save_predictions=True,
                         outfile_pref=outfile_pref+".val")

    test_mae, test_rmse = evaluate_loss(best_model, dataset, dataset.test,
                                        save_predictions=True,
                                        outfile_pref=outfile_pref+".test")

    logging.info("best-train-mae {:.5f}".format(train_mae))
    logging.info("best-val-mae {:.5f} "
                 "test-mae {:.5f} "
                 "test-rmse {:.5f}".format(val_mae, test_mae, test_rmse))

    if args.write_embeddings:
        model.write_embeddings(outfile_pref)
