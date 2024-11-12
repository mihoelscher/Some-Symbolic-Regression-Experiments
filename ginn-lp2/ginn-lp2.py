from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y
import numpy as np

class GINNLP2(BaseEstimator, RegressorMixin):

    # def __init__(self, max_ln_blocks, max_line_blocks, reg_change=0.3,
    #              l1_reg=0.1, l2_reg=0.01, num_epochs=200,
    #              round_digits=3):
    def __init__(self, reg_change=0.3, start_ln_blocks=1, growth_steps=2,
                 l1_reg=0.1, l2_reg=0.01, num_epochs=200,
                 round_digits=3, init_lr=0.01, decay_steps=1000,
                 train_iter=3):
        self.reg_change = reg_change
        # self.max_ln_blocks = max_ln_blocks
        # self.max_line_blocks = max_line_blocks
        self.start_ln_blocks = start_ln_blocks
        self.growth_steps = growth_steps
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.num_epochs = num_epochs
        self.round_digits = round_digits
        self.init_lr = init_lr
        self.decay_steps = decay_steps
        self.reg_change = reg_change
        self.train_iter = train_iter

    def fit(self, X, y):
        print(len(y))
        print(X, y)
        X, y = check_X_y(X, y)
        X = np.array(X).transpose()
        x_dim = X.shape[0]
        # model, train_history, blk_count, line_blk_count = train_model_growth_v3(
        #     X, y, self.max_ln_blocks, self.max_line_blocks, self.num_epochs,
        #     l1_reg=self.l1_reg, l2_reg=self.l2_reg)
        # model, train_history, blk_count, _ = train_model_growth(X, y, start_ln_block=self.start_ln_blocks,
        #                                                      num_epochs=self.num_epochs,
        #                                                      growth_steps=self.growth_steps, l1_reg=self.l1_reg,
        #                                                      l2_reg=self.l2_reg,
        #                                                      init_lr=self.init_lr,
        #                                                      decay_steps=self.decay_steps,
        #                                                      reg_change=self.reg_change)
        model, train_history, blk_count, model_eq, _ = select_best_model(X, y, start_ln_block=self.start_ln_blocks,
                                                                         num_epochs=self.num_epochs,
                                                                         growth_steps=self.growth_steps,
                                                                         l1_reg=self.l1_reg,
                                                                         l2_reg=self.l2_reg,
                                                                         init_lr=self.init_lr,
                                                                         decay_steps=self.decay_steps,
                                                                         reg_change=self.reg_change,
                                                                         num_iter=self.train_iter,
                                                                         round_digits=self.round_digits)
        self.model = model
        self.train_history = train_history
        self.blk_count = blk_count

        # recovered_eq = get_sympy_expr_v3(model, x_dim, blk_count, line_blk_count,
        #                                  round_digits=self.round_digits)
        self.recovered_eq = model_eq
        return self

    def predict(self, X):
        X = np.array(X).transpose()
        train_x = preprocess_data(X, is_train=False)
        pred_y = self.model.predict(train_x)
        print(pred_y)
        return pred_y