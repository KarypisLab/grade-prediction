import argparse
import torch
import collections as cl
import sys
from torch.utils.data import Dataset
import math
from util import readlines

class CCKRM_Data(Dataset):
    def __init__(self, args, mode='nocontext'):

        infile_pref            = args.infile_pref
        self.mode              = mode
        self.nprior            = args.nprior
        self.min_est_count     = args.min_est_count
        self.row_center_grades = args.row_center_grades
        try:
            self.apply_decay      = args.apply_decay
            self.lamda            = args.lamda
        except:
            self.apply_decay = 0
            self.lamda       = 0.0
        self.students = dict()
        self.courses = dict()
        self.target_course_freq = cl.defaultdict(int)
        self.prior_course_freq  = cl.defaultdict(int)
        self.concur_course_freq  = cl.defaultdict(int)
        self.max_student_size = 0
        self.max_prior_size, self.max_target_size = 0, 0
        self.max_concur_size = 0
        self.max_n_prior, self.max_n_concur = 0, 0
        self.npriors = list() # list of number of prior courses for all instances in the data

        self.train, self.students, self.courses = self.load_data(infile_pref+".train", train=True)

        self.val, _, _   = self.load_data(infile_pref+".val")
        self.test, _, _  = self.load_data(infile_pref+".test")

        self.num_students, self.num_courses = len(self.students), len(self.courses)


    def __len__(self):
        return len(self.train)

    def __getitem__(self, idx):
        assert idx < len(self.train), "Inside __get_item__: idx {} out of range!".format(idx)

        return self.prepare_sample(self.train[idx])

    def load_data(self, infile, train=False):

        students = dict()
        courses = dict()
        old_data = cl.defaultdict(set) # use this to check whether tid belongs to current (train, val, or test) set or previous (train or val) set

        for line in readlines(infile):
            try:
                sid, cid, tid, grade = line.split(" ")
            except:
                print("Wrong input file format!")
                sys.exit(1)

            sid = int(sid)
            cid = int(cid)
            tid = int(tid)
            grade = float(grade)

            if train:
                courses[cid] = True

            else: # val or test set
                if not self.row_center_grades and sid not in self.students:
                    continue
                if cid not in self.courses:
                    continue
                # copy training data for val student, training+val data for test student
                if sid in self.students:
                    students[sid] = self.students[sid]
                    old_data[sid].add(tid)

            students.setdefault(sid, dict())
            students[sid].setdefault(tid, dict())
            students[sid][tid][cid] = grade

        return self._construct_prior_target_data(students, old_data, train=train), students, courses

    def _construct_prior_target_data(self, students, old_data, train=False):

        data = list()

        for sid in students:
            prior_courses = dict()

            if train:
                self.max_student_size = max([self.max_student_size, sid+1])

            if self.row_center_grades:
                avg_prev_grade = self._compute_avg_grade(students[sid])
            else:
                avg_prev_grade = 0.0

            for tid in sorted(students[sid]):
                term_courses = students[sid][tid]


                if old_data and tid in old_data[sid]:
                    # add current term courses to prior courses
                    self._add_prior_courses(prior_courses,
                                            term_courses, tid,
                                            avg_prev_grade=avg_prev_grade,
                                            train=train)
                    continue

                if len(prior_courses) >= self.nprior: # construct a (prev, target) tuple
                    prior_cids, prior_grades = self._compute_prior_cids_grades(prior_courses,
                                                                               tid)

                    for cid in term_courses:
                        grade = term_courses[cid]
                        if train:
                            self.target_course_freq[cid] += 1
                            self.max_target_size = max([self.max_target_size, cid+1])
                        else: # make sure cid exists in training set with sufficient frequency
                            if cid not in self.target_course_freq or \
                               self.target_course_freq[cid] < self.min_est_count:
                                continue

                        concur_courses = self._add_concurrent_courses(cid,
                                                                      term_courses,
                                                                      train=train)
                        if len(concur_courses) > 0:
                            datum = {'sid': [sid],
                                     'prior_cids': prior_cids,
                                     'prior_grades': prior_grades,
                                     'concur_cids': concur_courses,
                                     'target_cid': [cid],
                                     'target_grade': [grade-avg_prev_grade],
                                     'avg_prev_grade': [avg_prev_grade]}
                            data.append(datum)
                            self.npriors.append(len(prior_cids))

                # add current term courses to prior courses
                self._add_prior_courses(prior_courses,
                                        term_courses, tid,
                                        avg_prev_grade=avg_prev_grade,
                                        train=train)

        return data

    def _compute_avg_grade(self, student):

        sum_grades = 0.0
        num_grades = 0

        for tid in student:
            for cid in student[tid]:
                grade = student[tid][cid]
                sum_grades += grade
                num_grades += 1

        return sum_grades/num_grades

    def _add_concurrent_courses(self, target_cid, courses, train=False):

        concur_courses = list()

        for cur_cid in courses:
            if cur_cid == target_cid:
                continue
            if train:
                self.concur_course_freq[cur_cid] += 1
                self.max_concur_size = max([self.max_concur_size, cur_cid+2]) # since we add a padding_idx of 0 for prior course embedding
            else: # make sure cid exists in training set with sufficient frequency
                if cur_cid not in self.concur_course_freq or \
                   self.concur_course_freq[cur_cid] < self.min_est_count:
                    continue
            concur_courses.append(cur_cid+1)

        self.max_n_concur = max(self.max_n_concur, len(concur_courses))

        return concur_courses

    # add current courses to previous courses
    def _add_prior_courses(self, prior_courses, term_courses, tid,
                           avg_prev_grade=0.0,
                           train=False):

        for cid in term_courses:
            grade = term_courses[cid]
            if grade == 0.0: # F grades; will not contribute to the student's knowledge state
                continue
            grade -= avg_prev_grade
            if grade == 0.0: # student has same grade in all prior courses
                grade = 0.01
            if train:
                self.prior_course_freq[cid] += 1
                self.max_prior_size = max([self.max_prior_size, cid+2]) # since we add a padding_idx of 0 for prior course embedding
            else: # make sure cid exists in training set with sufficient frequency
                if cid not in self.prior_course_freq or \
                   self.prior_course_freq[cid] < self.min_est_count:
                    continue
            prior_courses[cid+1] = [tid, grade] # since we add a padding_idx of 0 for prior course embedding

        self.max_n_prior = max(self.max_n_prior, len(prior_courses))

        return

    def _compute_prior_cids_grades(self, prior_courses, cur_tid, train=False):
        cids, grades = [], []
        for prior_cid in prior_courses:
            prior_tid, prior_grade = prior_courses[prior_cid]
            if self.apply_decay:
                decay = math.exp(-self.lamda * (cur_tid-prior_tid))
                prior_grade *= decay

            cids.append(prior_cid)
            grades.append(prior_grade)

        return cids, grades

    def _getitems(self, data):

        samples = list()

        for idx in range(len(data)):
            samples.append(self.prepare_sample(data[idx]))

        return samples

    def prepare_sample(self, datum):

        prior_cids   = datum['prior_cids']
        prior_grades = datum['prior_grades']
        concur_cids  = datum['concur_cids']

        # mask prior cids and grades to uniformalize their length
        mask_len = self.max_n_prior-len(prior_cids)
        prior_cids = prior_cids + [0]*mask_len
        prior_grades = prior_grades + [0]*mask_len

        # mask concurrent cids to uniformalize their length
        mask_len = self.max_n_concur-len(concur_cids)
        concur_cids = concur_cids + [0]*mask_len

        sample = {'sid': torch.LongTensor(datum['sid']),
                  'prior_cids': torch.LongTensor(prior_cids),
                  'prior_grades': torch.FloatTensor(prior_grades),
                  'concur_cids': torch.LongTensor(concur_cids),
                  'target_cid': torch.LongTensor(datum['target_cid']),
                  'target_grade': torch.FloatTensor(datum['target_grade']),
                  'avg_prev_grade': torch.FloatTensor(datum['avg_prev_grade'])}

        return sample

def parse_args():
    argparser = argparse.ArgumentParser(description="Loads dataset")

    argparser.add_argument("infile_pref", help="Path prefix for input files ", type=str)
    argparser.add_argument("--nprior", default=4, type=int, help="Min # prior courses for predicting a target course's grade. Default=4")
    argparser.add_argument("--min_est_count", default=10, type=int,
                           help="Min frequency of a course in the training set to be considered in the validation or test set. Default=10")
    argparser.add_argument("--apply_decay", default=0, type=int, choices=[0, 1],
                           help="Whether to apply decay on prior courses wrt time. Default=0")
    argparser.add_argument("--mode", choices=['context', 'nocontext'], type=str,
                           default='nocontext', help='Default=nocontext')
    argparser.add_argument("--lamda", default=0, type=float,
                           help="Decay constant on prior grades (if apply_decay=1). Default=0")

    return argparser.parse_args()

def test():
    datadir = '../data'
    data = InputData('{}/students'.format(datadir))

    print(data[0])
    print(data.val[0])

if __name__ == '__main__':

    args = parse_args()
    dataset = InputData(args, mode='nocontext')
