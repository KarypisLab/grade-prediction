# grade-prediction
Neural network models for grade prediction

This repository contains the implementation of the neural attentive
knowledge-based models developed in the following papers:

Sara Morsy and George Karypis, "Sparse Neural Attentive Knowledge-based Models
for Grade Prediction", in Proceedings of 12th International Conference on
Educational Data Mining (EDM), 2019.

Sara Morsy and George Karypis, "Context-aware Nonlinear and Neural Attentive Knowledge-based
Models for Grade Prediction", to appear in Journal of Educational Data Mining (JEDM), 2020. 

It uses python 2.7 and pytorch 0.3.1.

There are two main scripts to run: CKRM (for non-context-aware models) and CCKRM (for context-aware models).

A sample command line for running the code is as follows:

python CKRM.py --nprior 4 --min_est_count 10 --embedding_size 8 --attn_weight_size 4 --train_loss 0 --l2_reg 1e-05 --lrn_rate 0.007 --row_center_grades 1 --batch_size 1000 --epochs 100 --beta 1.0 --accumulate_priors 2 --sparsemax 1 --grade_b4_attn 1 --verbose 1 data/sample data

Input files should be of the format: 
  <student-id> <course-id> <term-id> <grade>
where <student-id> and <course-id> start from 1, <term-id> starts from 1 and denotes the term number when the student took the course, and <grade> is a numeric value.

The <data> folder should contain 3 files, one for training data, one for validation data, and one for testing data.
