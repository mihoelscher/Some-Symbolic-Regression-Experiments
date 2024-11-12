from sympy import symbols, powsimp, ln, preorder_traversal, Float, sqrt, count_ops, simplify


def select_best_model(train_x, train_y, start_ln_block, num_epochs, growth_steps=2,
                      init_lr=0.01, decay_steps=1000,
                      reg_change=0.3, l1_reg=1e-2, l2_reg=1e-2, num_iter=3,
                      round_digits=3, complexity_weight=1e-6):
    best_mse = float('inf')
    best_err = float('inf')
    best_model = None
    best_train_hist = None
    best_blk = None
    best_eq = None
    model_eq_list = []
    for i in range(num_iter):
        model, train_history, actual_blocks, mse_cur = train_model_growth(
            train_x, train_y, start_ln_block, num_epochs, growth_steps,
            init_lr, decay_steps, reg_change, l1_reg, l2_reg)
        model_eq = get_sympy_expr_v2(model, train_x.shape[0], actual_blocks, round_digits)
        model_complexity = eq_complexity(model_eq)
        model_err = mse_cur + model_complexity * complexity_weight
        model_eq_list.append(model_eq)
        if model_err < best_err:
            best_err = model_err
            best_mse = mse_cur
            best_model = model
            best_train_hist = train_history
            best_blk = actual_blocks
            best_eq = model_eq
    print(model_eq_list)
    return best_model, best_train_hist, best_blk, best_eq, best_mse

def train_model_growth(train_x, train_y, start_ln_block, num_epochs, growth_steps=2,
                       init_lr=0.01, decay_steps=1000,
                       reg_change=0.3, l1_reg=1e-2, l2_reg=1e-2):
    # train_x = [np.array(x) for x in np.array(train_x.transpose())]
    print(train_x)
    x_dim = len(train_x)
    cur_ln_blocks = start_ln_block
    model = eql_model_v2(x_dim, ln_block_count=start_ln_block, decay_steps=decay_steps)
    train_history = []
    print('model weights', model.get_weights())
    opt = eql_opt(decay_steps=decay_steps, init_lr=init_lr)
    model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mean_squared_error'])
    # train_history.append(
    #     model.fit(train_x, train_y, epochs=num_epochs, batch_size=32, validation_split=0.2)
    # )

    model, history, mse = reg_stages_train(model, train_x, train_y, num_epochs, reg_change=reg_change,
                                           l1_reg=l1_reg, l2_reg=l2_reg)
    mse_cur = mse
    print('MSE', mse)
    actual_blocks = start_ln_block
    train_history += history
    for i in range(growth_steps):
        print(model.get_weights())
        tf.keras.utils.get_custom_objects().update({'log_activation': log_activation,
                                                    'L1L2_m': L1L2_m})
        model_new = tf.keras.models.clone_model(model)
        model_new.set_weights(model.get_weights())
        model_new = add_ln_block(x_dim, model_new, cur_ln_blocks)
        print(model_new.get_weights())
        cur_ln_blocks += 1
        opt = eql_opt(decay_steps=decay_steps, init_lr=init_lr)
        model_new.compile(optimizer=opt, loss='mean_squared_error', metrics=['mean_squared_error'])
        # train_history.append(
        #     model.fit(train_x, train_y, epochs=num_epochs, batch_size=32, validation_split=0.2)
        # )
        model_new, history, mse = reg_stages_train(model_new, train_x, train_y, num_epochs,
                                                   l1_reg=l1_reg, l2_reg=l2_reg,
                                                   reg_change=reg_change)
        print('MSE', mse)
        if mse > mse_cur * 0.8 and mse_cur < 1e-4:
            break
        model = tf.keras.models.clone_model(model_new)
        model.set_weights(model_new.get_weights())
        actual_blocks += 1
        mse_cur = mse
        train_history += history
    print(model.get_weights())
    return model, train_history, actual_blocks, mse_cur




### formula utils ###
def get_sympy_expr_v2(model, x_dim, ln_layer_count=2, round_digits=3):
    sym = []
    for i in range(x_dim):
        sym.append(symbols('X_' + str(i), positive=True))

    ln_dense_weights = [model.get_layer('ln_dense_{}'.format(i)).get_weights() for i in range(ln_layer_count)]
    output_dense_weights = model.get_layer('output_dense').get_weights()
    print(ln_dense_weights)
    print(output_dense_weights)
    sym_expr = 0
    for i in range(ln_layer_count):
        ln_block_expr = output_dense_weights[0][i][0]
        for j in range(x_dim):
            ln_block_expr *= sym[j] ** ln_dense_weights[i][0][j][0]
        sym_expr += ln_block_expr
    sym_expr = round_sympy_expr(sym_expr, round_digits=round_digits)

    print(sym_expr)

    return sym_expr

def round_sympy_expr(sym_expr, round_digits=2):
    for a in preorder_traversal(sym_expr):
        if isinstance(a, Float):
            if int(round(a, round_digits)) == round(a, round_digits):
                sym_expr = sym_expr.subs(a, int(round(a, round_digits)))
            else:
                sym_expr = sym_expr.subs(a, round(a, round_digits))
    return sym_expr

def eq_complexity(expr):
    c = 0
    for arg in preorder_traversal(expr):
        c += 1
    return c