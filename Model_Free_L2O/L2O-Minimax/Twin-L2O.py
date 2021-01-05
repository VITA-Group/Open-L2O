
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import multiprocessing
import os
import csv
import copy
import joblib
from torchvision import datasets
import torchvision
import json
import numpy as np
import math
import argparse
import seaborn as sns
import datetime
from torchviz import make_dot
from graphviz import Digraph
import torch
from torch.autograd import Variable
from torchviz import make_dot
import sys
sys.path.insert(0, './utils')
import metrics
import pandas as pd
datetime.datetime.now()

USE_CUDA = torch.cuda.is_available()

if not os.path.exists("../result"):
    os.mkdir("../result")

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="saddle_train", type=str)
parser.add_argument("--eval_interval",type=int, default=5)

args = parser.parse_args()
config = args.config
config = args.config

with open("config/" + config + ".json") as temp:
    params = json.load(temp)

save_name = "../../result/" + params["save_name"] 

if params['save_name'] and not os.path.exists(save_name):
    os.makedirs(save_name)

load_name = "../../result/" + params["load_name"] 


torch.manual_seed(params['seed'])
args.rescale = params['rescale']

def detach_var(v):
    var = w(Variable(v.data, requires_grad=True))
    var.retain_grad()
    return var


def w(v):
    if USE_CUDA:
        return v.cuda()
    return v

def sche_ratio(epoch_index):
    if 0 <= epoch_index <= 80:
        return 0.2 + epoch_index/100
    else:
        return 1
  

def do_fit(opt_net_min, opt_net_max, meta_opt_min, meta_opt_max, target_loss, target_optimizee, unroll_unit, optim_it,
           should_train,data,
           out_mul=1, epoch_index=None, iteration_index=None, batch_size=1):
    sche_lr = list(np.arange(1e-4, 1e-1, (1e-1 - 1e-4) / int(0.2 * optim_it)))
    if should_train:
        opt_net_min.train()
        opt_net_max.train()

    else:
        opt_net_min.eval()
        opt_net_max.eval()

    batch_data = data[iteration_index * batch_size:(iteration_index + 1) * batch_size, :]
    batch_target = []
    batch_target.clear()
    batch_optimizee = []
    batch_optimizee.clear()
    for i_temp in range(batch_data.shape[0]):
        target_temp = target_loss(index=i_temp, data=batch_data, training=should_train)
        optimizee_temp = w(target_optimizee())
        batch_target.append(target_temp)
        batch_optimizee.append(optimizee_temp)

    hidden_states_min = [w(Variable(torch.zeros(batch_size, opt_net_min.hidden_sz))) for _ in range(2)]
    cell_states_min = [w(Variable(torch.zeros(batch_size, opt_net_min.hidden_sz))) for _ in range(2)]
    hidden_states_min = [torch.nn.init.normal_(hidden_state_min, mean=0.0, std=0.01) for hidden_state_min in
                         hidden_states_min]
    cell_states_min = [torch.nn.init.normal_(cell_state_min, mean=0.0, std=0.01) for cell_state_min in cell_states_min]

    hidden_states_max = [w(Variable(torch.zeros(batch_size, opt_net_max.hidden_sz))) for _ in range(2)]
    cell_states_max = [w(Variable(torch.zeros(batch_size, opt_net_max.hidden_sz))) for _ in range(2)]
    hidden_states_max = [torch.nn.init.normal_(hidden_state_max, mean=0.0, std=0.01) for hidden_state_max in
                         hidden_states_max]
    cell_states_max = [torch.nn.init.normal_(cell_state_max, mean=0.0, std=0.01) for cell_state_max in cell_states_max]

    all_distance_ever_u = [[]]
    all_distance_ever_v = [[]]
    for _ in range(batch_size - 1):
        all_distance_ever_u.append([])
        all_distance_ever_v.append([])
    if should_train:
        meta_opt_min.zero_grad()
        meta_opt_max.zero_grad()
    all_losses_min = None
    all_losses_max = None
    batch_reward_min = [None for _ in range(batch_size)]
    batch_reward_max = [None for _ in range(batch_size)]

    if params['mode'] == "test":
        curve_info = np.zeros((batch_size, optim_it//2+1, 2))
    else :    
        curve_info = np.zeros((batch_size, optim_it, 2))
    if params['mode'] == "test":
        for i_temp in range(batch_size):
            if USE_CUDA:
                curve_info[i_temp,0,0] = batch_optimizee[i_temp].all_named_parameters()[0][1].data.cpu().numpy()[0] #store initial u  
                curve_info[i_temp,0,1] = batch_optimizee[i_temp].all_named_parameters()[1][1].data.cpu().numpy()[0] #store initial v
            else:
                curve_info[i_temp,0,0] = batch_optimizee[i_temp].all_named_parameters()[0][1].data.numpy()[0] #store initial u  
                curve_info[i_temp,0,1] = batch_optimizee[i_temp].all_named_parameters()[1][1].data.numpy()

    for iteration in range(1, optim_it + 1):
        t1 = datetime.datetime.now()

        offset = 0
        result_params = {"theta_min": [], "theta_max": []}

        if not iteration % (unroll_unit * 2) == 0:
            hidden_states2_min = [w(Variable(torch.zeros(batch_size, opt_net_min.hidden_sz))) for _ in range(2)]
            cell_states2_min = [w(Variable(torch.zeros(batch_size, opt_net_min.hidden_sz))) for _ in range(2)]
        hidden_states2_max = [w(Variable(torch.zeros(batch_size, opt_net_max.hidden_sz))) for _ in range(2)]
        cell_states2_max = [w(Variable(torch.zeros(batch_size, opt_net_max.hidden_sz))) for _ in range(2)]

        batch_gradient = []
        batch_gradient.clear()

        if iteration % 2 == 1:  # min  step
            batch_input_result = []
            for i_temp, optimizee_temp in enumerate(batch_optimizee):
                temp_v = batch_optimizee[i_temp].all_named_parameters()[1][1].clone().detach()
                temp_u = batch_optimizee[i_temp].all_named_parameters()[0][1].clone().detach()
                p = batch_target[i_temp].get_grad_u(theta_min=temp_u, theta_max=temp_v)  # grad_u
                batch_gradient.append(p)
                cur_sz = batch_size
                input_temp = batch_target[i_temp].get_grad_v(theta_min=temp_u, theta_max=temp_v)  # grad_u
                batch_input_result.append(input_temp)

            batch_gradient_result = w(torch.tensor(batch_gradient).view(batch_size, 1))
            batch_input_result = w(torch.tensor(batch_input_result).view(batch_size, 1))

            alpha = args.rescale
            hidden_states_min = [alpha*h for h in hidden_states_min]
            cell_states_min = [alpha * h for h in cell_states_min]
            # rescaling the hidden states to allevitate potential diverging outcome

            updates, new_hidden, new_cell = opt_net_min(
                batch_gradient_result, batch_input_result,
                [h[offset:offset + cur_sz] for h in hidden_states_min],
                [c[offset:offset + cur_sz] for c in cell_states_min])
            for i in range(len(new_hidden)):
                hidden_states2_min[i][offset:offset + cur_sz] = new_hidden[i]
                cell_states2_min[i][offset:offset + cur_sz] = new_cell[i]

            for i_temp, optimizee_temp in enumerate(batch_optimizee):
                if iteration < 0.2 * optim_it:
                    temp = optimizee_temp.all_named_parameters()[0][1] + torch.sign(
                        updates[i_temp].view(*optimizee_temp.all_named_parameters()[0][1].size())) * sche_lr[
                               iteration - 1]
                else:
                    temp = optimizee_temp.all_named_parameters()[0][1] + updates[i_temp].view(
                        *optimizee_temp.all_named_parameters()[0][1].size()) * out_mul

                result_params["theta_min"].append(temp)
                result_params["theta_min"][i_temp].retain_grad()
                result_params["theta_max"].append(optimizee_temp.all_named_parameters()[1][1])

        if iteration % 2 == 0:  # max  step
            batch_input_result = []
            for i_temp, optimizee_temp in enumerate(batch_optimizee):
                temp_v = batch_optimizee[i_temp].all_named_parameters()[1][1].clone().detach()
                temp_u = batch_optimizee[i_temp].all_named_parameters()[0][1].clone().detach()
                p = batch_target[i_temp].get_grad_v(theta_min=temp_u, theta_max=temp_v)  # grad_v
                batch_gradient.append(p)
                cur_sz = batch_size
                input_temp = batch_target[i_temp].get_grad_u(theta_min=temp_u, theta_max=temp_v)  # grad_u
                batch_input_result.append(input_temp)
            batch_gradient_result = w(torch.tensor(batch_gradient).view(batch_size, 1))
            batch_input_result = w(torch.tensor(batch_input_result).view(batch_size, 1))

            alpha = args.rescale
            hidden_states_max = [alpha*h for h in hidden_states_max]
            cell_states_max = [alpha * h for h in cell_states_max]

            updates, new_hidden, new_cell = opt_net_max(
                batch_gradient_result, batch_input_result,
                [h[offset:offset + cur_sz] for h in hidden_states_max],
                [c[offset:offset + cur_sz] for c in cell_states_max])
            for i in range(len(new_hidden)):
                hidden_states2_max[i][offset:offset + cur_sz] = new_hidden[i]
                cell_states2_max[i][offset:offset + cur_sz] = new_cell[i]

            for i_temp, optimizee_temp in enumerate(batch_optimizee):
                if iteration < 0.2 * optim_it:
                    temp = optimizee_temp.all_named_parameters()[1][1] + torch.sign(
                        updates[i_temp].view(*optimizee_temp.all_named_parameters()[1][1].size())) * sche_lr[
                               iteration - 1]
                else:
                    temp = optimizee_temp.all_named_parameters()[1][1] + updates[i_temp].view(
                        *optimizee_temp.all_named_parameters()[1][1].size()) * out_mul

                result_params["theta_max"].append(temp)
                result_params["theta_max"][i_temp].retain_grad()
                result_params["theta_min"].append(optimizee_temp.all_named_parameters()[0][1])

        if iteration % (unroll_unit * 2) == 0: 
            if should_train:
                meta_opt_min.zero_grad()
                meta_opt_max.zero_grad()
                if "CL" in params and params['CL']:  ## Curriculum learning
                    count_temp = 0
                    index = metrics.get_index(curve_info, iteration_index=iteration,unroll_unit=unroll_unit,batch_data=batch_data,
                                                ratio=sche_ratio(epoch_index))
                    for b_temp in index:
                        count_temp += 1
                        if all_losses_min is None:
                            all_losses_min = batch_reward_min[b_temp]
                        else:
                            all_losses_min += batch_reward_min[b_temp]
                        if all_losses_max is None:
                            all_losses_max = batch_reward_max[b_temp]
                        else:
                            all_losses_max += batch_reward_max[b_temp]
                else:  ## normal training
                    for b_temp in range(batch_size):
                        if all_losses_min is None:
                            all_losses_min = batch_reward_min[b_temp]
                        else:
                            all_losses_min += batch_reward_min[b_temp]
                        if all_losses_max is None:
                            all_losses_max = batch_reward_max[b_temp]
                        else:
                            all_losses_max += batch_reward_max[b_temp]
                if "CL" in params and params['CL']:
                    print( "count of instances that put in reward of LSTM", count_temp)
                    if count_temp > 0:
                        total_loss = all_losses_max + all_losses_min
                        total_loss.backward()
                        meta_opt_min.step()
                        meta_opt_max.step()

                else:
                    total_loss = all_losses_max + all_losses_min
                    total_loss.backward()
                    meta_opt_min.step()
                    meta_opt_max.step()

            all_losses_min = None
            all_losses_max = None
            batch_reward_min = [None for _ in range(batch_size)]
            batch_reward_max = [None for _ in range(batch_size)]
            batch_optimizee = []
            for i_temp in range(batch_size):
                batch_optimizee.append(w(target_optimizee(theta_min=detach_var(result_params["theta_min"][i_temp]),
                                                          theta_max=detach_var(result_params["theta_max"][i_temp]))))

            hidden_states_min = [detach_var(v) for v in hidden_states2_min]
            cell_states_min = [detach_var(v) for v in cell_states2_min]

            hidden_states_max = [detach_var(v) for v in hidden_states2_max]
            cell_states_max = [detach_var(v) for v in cell_states2_max]
            loss_previous = [0.0 for _ in loss_previous]  # modify here

        else:
            batch_optimizee = []
            batch_optimizee.clear()
            for i_temp in range(batch_size):
                batch_optimizee.append(w(target_optimizee(theta_min=result_params["theta_min"][i_temp],
                                                          theta_max=result_params["theta_max"][i_temp])))

            if iteration % 2 == 1:  # min step -- update u
                hidden_states_min = hidden_states2_min
                cell_states_min = cell_states2_min
                if (hidden_states_min[0] == 0.0).all():
                    assert False
            else:  # max step -- update v
                hidden_states_max = hidden_states2_max
                cell_states_max = cell_states2_max
                if (hidden_states_max[0] == 0.0).all():
                    assert False

        ############################## Get current solution and save #######################
        ####################################### Define Evaluation #########################################
        for i_temp in range(batch_size):
            solution = [0, 0]
            for i, p in enumerate(batch_optimizee[i_temp].all_named_parameters()):
                if USE_CUDA:
                    solution[i] = p[1].data.cpu().numpy()[0]
                else:
                    solution[i] = p[1].data.numpy()[0]

            if params['mode'] == "test":
                if iteration % 2 == 1: # min step:
                    curve_info[i_temp, iteration//2 + 1, 0] = solution[0]
                elif iteration % 2 == 0: # max step:
                    curve_info[i_temp, iteration//2, 1] = solution[1]
            else:            
                curve_info[i_temp, iteration - 1, 0] = solution[0]
                curve_info[i_temp, iteration - 1, 1] = solution[1]
            if params['loss'] == 1:
                all_distance_ever_v[i_temp].append(abs(solution[1]))
                all_distance_ever_u[i_temp].append(abs(solution[0]))
            if params['loss'] == 2:
                all_distance_ever_v[i_temp].append(abs(solution[1]))
                all_distance_ever_u[i_temp].append(abs(solution[0]))
            if params['loss'] == 3:

                best_dist_u = 100000
                initial_dist = abs(solution[0])
                k = 0
                while True:
                    gt = k / batch_target[i_temp].a
                    if USE_CUDA:
                        gt = gt.cpu().data.numpy()[0]
                    else:
                        gt = gt.cpu().data.numpy()[0]

                    dist = abs(abs(solution[0]) - gt)
                    if dist < best_dist_u:
                        best_dist_u = dist
                    elif dist > initial_dist:
                        break
                    k = k + 1
                all_distance_ever_v[i_temp].append(abs(solution[1]))
                all_distance_ever_u[i_temp].append(best_dist_u)

        ######*********************** define reward ***********************####
        if iteration == 1:
            loss_previous = []
            for _ in range(batch_size):
                loss_previous.append(0)

        for i_temp in range(batch_size):
            if iteration % 2 == 1:  # min step
                loss_temp = batch_optimizee[i_temp](batch_target[i_temp])

            elif iteration % 2 == 0:  # max step
                loss_temp = (-1) * batch_optimizee[i_temp](batch_target[i_temp])

            if iteration % 2 == 1:  # min step
                if batch_reward_min[i_temp] is None:
                    batch_reward_min[i_temp] = (loss_temp + loss_previous[i_temp])  / batch_size
                else:
                    batch_reward_min[i_temp] += (loss_temp + loss_previous[i_temp]) / batch_size
            elif iteration % 2 == 0:  # max step
                if batch_reward_max[i_temp] is None:
                    batch_reward_max[i_temp] = (loss_temp + loss_previous[i_temp])  / batch_size
                else:
                    batch_reward_max[i_temp] += (loss_temp + loss_previous[i_temp])  / batch_size

            loss_previous[i_temp] = loss_temp.clone() #modify


        ######*********************** freezing opposite variable***********************####
        batch_theta_min_temp = []
        batch_theta_max_temp = []

        if iteration % 2 == 1:  # min step
            for optimizee_temp in batch_optimizee:
                batch_theta_min_temp.append(optimizee_temp.theta_min.detach())
                batch_theta_max_temp.append(optimizee_temp.theta_max)

            batch_optimizee = []
            for i_temp in range(batch_size):
                batch_theta_max_temp[i_temp].requires_grad = True
                batch_optimizee.append(w(target_optimizee(batch_theta_min_temp[i_temp], batch_theta_max_temp[i_temp])))
        if iteration % 2 == 0:  # max step
            for optimizee_temp in batch_optimizee:
                batch_theta_max_temp.append(optimizee_temp.theta_max.detach())
                batch_theta_min_temp.append(optimizee_temp.theta_min)

            batch_optimizee = []
            for i_temp in range(batch_size):
                batch_theta_min_temp[i_temp].requires_grad = True
                batch_optimizee.append(w(target_optimizee(batch_theta_min_temp[i_temp], batch_theta_max_temp[i_temp])))


        ################################ save curve info ################################
    if should_train and params['save_curve']:
        save_name = "../../result/" + params["save_name"]

        if not os.path.exists(save_name):
            save_name = load_name
        if not os.path.exists(save_name + "/" + "curve_info/" + "epoch" + str(epoch_index) + "/"):
            os.makedirs(save_name + "/" + "curve_info/" + "epoch" + str(epoch_index) + "/")

        for i_temp in range(batch_size):
            temp_name = "_u_" + str(round(batch_target[i_temp].a.item(), 3)) + "_v_" + str(
                round(batch_target[i_temp].b.item(), 3))
            np.savetxt(save_name + "/" + "curve_info/" + "epoch" + str(epoch_index) + "/iteration" + str(
                iteration_index) + "_batch_index_" + str(i_temp) + temp_name + ".txt", curve_info[i_temp, :, :])
    elif params['mode'] == "test" and params['save_curve']:

        save_name = "../../result/" + params["save_name"]

        if not os.path.exists(save_name):
            save_name = load_name
        if not os.path.exists(save_name + "/" + "curve_info/" + "test_example" + "/" + params['load_model']):
            os.makedirs(save_name + "/" + "curve_info/" + "test_example/" + params['load_model'] + "/")
        for i_temp in range(batch_size):
            temp_name = "_u_" + str(round(batch_target[i_temp].a.item(), 3)) + "_v_" + str(
                round(batch_target[i_temp].b.item(), 3))
            np.savetxt(
                save_name + "/" + "curve_info/" + "test_example/" + params['load_model'] + "/" + "sample_index_" + str(
                    iteration_index) + "_epoch_" + str(epoch_index) + temp_name + ".txt", curve_info[i_temp, :, :])

    t2 = datetime.datetime.now()
    print(t2 - t1)

    if params['mode'] == "test":
        param_a = batch_target[0].a
        param_b = batch_target[0].b
        return all_distance_ever_v, all_distance_ever_u, param_a, param_b

    else:
        return all_distance_ever_v, all_distance_ever_u


def fit_optimizer(target_loss, target_optimizee, n_train, batch_size, unroll_unit,
                  train_optim_it, eval_optim_it, n_epochs, n_eval, lr, save_name , preproc=False, out_mul=1):

    # Summary: for training the LSTM, we train it n_epochs,.
    # For each epoch, the training data -- target loss function size is 128
    # these 128 training data are different random loss functions
    # and the validation data are n_tests random functions
    # We pick the best LSTM according to the performance on the validation set
    opt_net_min = w(Optimizer(preproc=preproc))
    opt_net_max = w(Optimizer(preproc=preproc))


    meta_opt_min = optim.Adam(opt_net_min.parameters(), lr=lr)
    meta_opt_max = optim.Adam(opt_net_max.parameters(), lr=lr)

    best_loss = 100000000000000000
    fit_data_v_train = np.zeros((batch_size, train_optim_it, 1))
    fit_data_v_train_detail = np.zeros((1, train_optim_it, batch_size))
    fit_data_u_train = np.zeros((batch_size, train_optim_it, 1))
    fit_data_u_train_detail = np.zeros(
        (1, train_optim_it, batch_size))  # fit_data_eval = np.zeros((n_eval,eval_optim_it*n_epochs,1))
    training_data = np.loadtxt(params['train_data_path'])
    training_data = training_data[0:n_train, :]
    eval_data = np.loadtxt(params['eval_data_path'])
    eval_data = eval_data[0:n_eval, :]
    np.random.seed(10)
    for epoch_temp in range(n_epochs):

        np.random.shuffle(training_data)
        print("training ... new epoch", epoch_temp)
        for iteration_temp in range(n_train // batch_size):
            print("training L2O ... iteration", iteration_temp)
            t9 = datetime.datetime.now()
            temp_v, temp_u = do_fit(opt_net_min, opt_net_max, meta_opt_min, meta_opt_max, target_loss, target_optimizee,
                                    unroll_unit=unroll_unit,
                                    optim_it=train_optim_it,
                                    out_mul=out_mul, should_train=True,
                                    data=training_data,
                                    epoch_index=epoch_temp, iteration_index=iteration_temp, batch_size=batch_size)
            fit_data_v_train[:, :, 0] = np.array([temp_v])

            fit_data_u_train[:, :, 0] = np.array([temp_u])
            t10 = datetime.datetime.now()
            print("training cost time...", t10 - t9)
            for i_temp in range(batch_size):
                fit_data_v_train_detail[:, :, i_temp] = fit_data_v_train[i_temp, :, :].reshape((1, train_optim_it))
                fit_data_u_train_detail[:, :, i_temp] = fit_data_u_train[i_temp, :, :].reshape((1, train_optim_it))

            if params['save_train'] == 1:
                sns.set(color_codes=True)
                sns.set_style("white")

                fit_data_v_train_df = fit_data_v_train.reshape((fit_data_v_train.shape[0],fit_data_v_train.shape[1]))
                fit_data_v_train_df = pd.DataFrame(fit_data_v_train_df).melt()
                ax = sns.lineplot(x="variable", y="value", data=fit_data_v_train_df,linewidth=7)
                ax.lines[-1].set_linestyle('-')
                for i_temp in range(batch_size):
                    plt.plot(list(range(train_optim_it)), (fit_data_v_train[i_temp, :, :]), alpha=0.3, linewidth=2,
                             label=str(i_temp))
                ax.legend()
                # plt.yscale('log')
                plt.xlabel('iteration')
                plt.ylabel('training performance -- distance between solution and ground truth')
                if params['loss'] == 1:
                    plt.title("Toy Example:" + r'$au^2-bv^2$')
                elif params['loss'] == 2:
                    plt.title("Toy Example:" + r'$au^{2}-bv^{2}+2uv$')
                elif params['loss'] == 3:
                    plt.title("Toy Example:" + r'$-bv \sin (a\pi u)$')

                if not os.path.exists(save_name + "/TrainingPlot/"):
                    os.makedirs(save_name + "/TrainingPlot/")
                plt.savefig(save_name + "/TrainingPlot/" + "epoch" + str(epoch_temp) + "_iteration" + str(
                    iteration_temp) + "_v.pdf")
                plt.close()

                fit_data_u_train_df = fit_data_v_train.reshape((fit_data_u_train.shape[0],fit_data_u_train.shape[1]))
                fit_data_u_train_df = pd.DataFrame(fit_data_u_train_df).melt()
                ax = sns.lineplot(x="variable", y="value", data=fit_data_u_train_df,linewidth=7)

                ax.lines[-1].set_linestyle('-')
                for i_temp in range(batch_size):
                    plt.plot(list(range(train_optim_it)), (fit_data_u_train[i_temp, :, :]), alpha=0.3, linewidth=2,
                             label=str(i_temp))
                ax.legend()
                # plt.yscale('log')
                plt.xlabel('iteration')
                plt.ylabel('training performance -- distance between solution and ground truth')
                if params['loss'] == 1:
                    plt.title("Toy Example:" + r'$au^2-bv^2$')
                elif params['loss'] == 2:
                    plt.title("Toy Example:" + r'$au^{2}-bv^{2}+2uv$')
                elif params['loss'] == 3:
                    plt.title("Toy Example:" + r'$-bv \sin (a\pi u)$')

                if not os.path.exists(save_name + "/TrainingPlot/"):
                    os.makedirs(save_name + "/TrainingPlot/")
                plt.savefig(save_name + "/TrainingPlot/" + "epoch" + str(epoch_temp) + "_iteration" + str(
                    iteration_temp) + "_u.pdf")
                plt.close()
        if not os.path.exists(save_name + "/"):
            os.makedirs(save_name + "/")

        temp_net_min = copy.deepcopy(opt_net_min.state_dict())
        temp_net_max = copy.deepcopy(opt_net_max.state_dict())
        torch.save(temp_net_min, save_name + "/" + "epoch" + str(epoch_temp) + "min.pth")
        torch.save(temp_net_max, save_name + "/" + "epoch" + str(epoch_temp) + "max.pth")

        if epoch_temp % args.eval_interval == 0:
            eval_result = []
            for index_eval in range(n_eval):
                all_distance_v, all_distance_u = do_fit(opt_net_min, opt_net_max, meta_opt_min, meta_opt_max, target_loss, target_optimizee,
                       unroll_unit=unroll_unit, optim_it=eval_optim_it,
                       out_mul=out_mul,
                       should_train=False, data=eval_data, iteration_index=index_eval, batch_size=1)

                burning_ratio = 0.5
                burning_period = int(burning_ratio * params['eval_optim_iter'])
                all_distance_v = all_distance_v[0][burning_period:]
                all_distance_u = all_distance_u[0][burning_period:]
                eval_result.append(np.sum(np.add(np.square(all_distance_v),np.square(all_distance_u))))

            loss = np.mean(eval_result)


            print(loss)
            if loss < best_loss:
                print("best loss after burning period", best_loss, "loss", loss)
                best_loss = loss
                best_net_min = copy.deepcopy(opt_net_min.state_dict())
                best_net_max = copy.deepcopy(opt_net_max.state_dict())
                torch.save(best_net_min, save_name + "/" + "best" + "min.pth")
                torch.save(best_net_max, save_name + "/" + "best" + "max.pth")

    return best_loss, best_net_min, best_net_max


class ToyLoss:
    def __init__(self, index, data, **kwargs):
        self.a = w(Variable(torch.tensor([data[index, 0]])))
        self.b = w(Variable(torch.tensor([data[index, 1]])))

    def get_loss(self, theta_min, theta_max):
        if params["loss"] == 1:
            return self.a * theta_min * theta_min - self.b * theta_max * theta_max
        elif params['loss'] == 2:
            return self.a * theta_min * theta_min - self.b * theta_max * theta_max + 2 * theta_max * theta_min
        elif params['loss'] == 3:
            return (-1) * self.b * theta_max * torch.sin(math.pi * theta_min * self.a)

    def get_grad_u(self, theta_min, theta_max):  # theta_max is v
        #
        if params['loss'] == 1:  # 2au
            return 2 * theta_min * self.a
        elif params['loss'] == 2:  # 2au+2v
            return 2 * theta_min * self.a + 2 * theta_max
        elif params['loss'] == 3:
            return (-1) * self.b * theta_max * torch.cos(
                self.a * math.pi * theta_min) * self.a * math.pi  # −bvcos(aπu)×aπ

    def get_grad_v(self, theta_min, theta_max):  # theta_min is u
        if params['loss'] == 1:  # −2bv
            return (-2) * self.b * theta_max
        elif params['loss'] == 2:  # −2bv+2u
            return (-2) * self.b * theta_max + 2 * theta_min
        elif params['loss'] == 3:  # −bsin(aπu)
            return (-1) * self.b * torch.sin(self.a * math.pi * theta_min)

class DualOptimizee(nn.Module):
    def __init__(self, theta_min=None, theta_max=None):
        super().__init__()
        # Note: assuming the same optimization for theta as for
        # the function to find out itself.
        if theta_min is None:
            self.theta_min = nn.Parameter(0.5 * ((-1) + 2 * torch.rand(1)))
        if theta_max is None:
            self.theta_max = nn.Parameter(0.5 * ((-1) + 2 * torch.rand(1)))

        if not theta_min is None:
            self.theta_min = theta_min
        if not theta_max is None:
            self.theta_max = theta_max

    def forward(self, target):
        return target.get_loss(self.theta_min, self.theta_max)

    def all_named_parameters(self):
        return [('theta_min', self.theta_min), ('theta_max', self.theta_max)]


class Optimizer(nn.Module):
    def __init__(self, preproc=False, hidden_sz=params['hidden_size'], preproc_factor=10.0):
        super().__init__()
        self.hidden_sz = hidden_sz
        if params['use_second']: # LSTM takes two inputs
            self.recurs = nn.LSTMCell(2, hidden_sz)
            # nn.LSTMCell (input_size, hidden_size)
            # output_size: (h_1, c_1); h_1 of shape (batch, hidden_size);c_1 of shape (batch, hidden_size)
            # here the batch = number of optimizee parameters
        else:
            self.recurs = nn.LSTMCell(1, hidden_sz)
        self.recurs2 = nn.LSTMCell(hidden_sz, hidden_sz)
        self.output = nn.Linear(hidden_sz, 1)
        self.preproc = preproc
        self.preproc_factor = preproc_factor
        self.preproc_threshold = np.exp(-preproc_factor)

    def forward(self, inp, second, hidden, cell):
        if self.preproc:
            # Implement preproc described in Appendix A

            # Note: we do all this work on tensors, which means
            # the gradients won't propagate through inp. This
            # should be ok because the algorithm involves
            # making sure that inp is already detached.
            inp = inp.data
            second = second.data
            inp2 = w(torch.zeros(inp.size()[0], 2))
            keep_grads = (torch.abs(inp) >= self.preproc_threshold).squeeze()
            inp2[:, 0][keep_grads] = (torch.log(torch.abs(inp[keep_grads]) + 1e-8) / self.preproc_factor).squeeze()
            inp2[:, 1][keep_grads] = torch.sign(inp[keep_grads]).squeeze()

            inp2[:, 0][~keep_grads] = -1
            inp2[:, 1][~keep_grads] = (float(np.exp(self.preproc_factor)) * inp[~keep_grads]).squeeze()
            inp = w(Variable(inp2))
        if params['use_second'] == 1:

            # hidden0, cell0 = self.recurs(w(torch.tensor(torch.cat((inp,second),1))), (hidden[0], cell[0]))
            hidden0, cell0 = self.recurs(w(torch.cat((inp, second), 1)), (hidden[0], cell[0]))
            hidden1, cell1 = self.recurs2(hidden0, (hidden[1], cell[1]))
        else:
            hidden0, cell0 = self.recurs(inp, (hidden[0], cell[0]))
            hidden1, cell1 = self.recurs2(hidden0, (hidden[1], cell[1]))
        return self.output(hidden1), (hidden0, hidden1), (cell0, cell1)


######################## Training
if __name__ == "__main__":

    if params['mode'] == "train":
        dist, optimizer_min, optimizer_max = fit_optimizer(target_loss=ToyLoss, target_optimizee=DualOptimizee,
                                                           lr=params['meta_lr'], n_epochs=params['train_epochs'],
                                                           n_train=params['n_train'],
                                                           unroll_unit=params['unroll_unit'],
                                                           train_optim_it=params['train_optim_iter'],
                                                           eval_optim_it=params['eval_optim_iter'],
                                                           n_eval=params['n_eval'],
                                                           save_name=save_name,
                                                           batch_size=params['batch_size']
                                                           )


    elif params['mode'] == "test":
        ##################### testing
        fit_data_v = np.zeros((params['test_epochs'], params['test_optim_iter'], 1))
        fit_data_u = np.zeros((params['test_epochs'], params['test_optim_iter'], 1))
        opt_min = w(Optimizer())
        opt_max = w(Optimizer())
        if USE_CUDA:
            opt_min.load_state_dict(
                torch.load(load_name + "/" + params['load_model'] + "min.pth", map_location=torch.device('cpu')))
            opt_max.load_state_dict(
                torch.load(load_name + "/" + params['load_model'] + "max.pth", map_location=torch.device('cpu')))
        else:
            opt_min.load_state_dict(torch.load(load_name + "/" + params['load_model'] + "min.pth"))
            opt_max.load_state_dict(torch.load(load_name + "/" + params['load_model'] + "max.pth"))

        test_data = np.loadtxt(params['test_data_path'])
        test_data = test_data[0:params['n_test'], :]
        torch.manual_seed(70)
        for j_temp in range(params['n_test']):
            for i_temp in range(params['test_epochs']):
                temp_v, temp_u, temp_a, temp_b = do_fit(opt_net_min=opt_min, opt_net_max=opt_max,
                                                        meta_opt_min=None, meta_opt_max=None,
                                                        target_loss=ToyLoss,
                                                        target_optimizee=DualOptimizee,
                                                        unroll_unit=params['unroll_unit'],
                                                        optim_it=params['test_optim_iter'],
                                                        out_mul=1.0,
                                                        should_train=False, data=test_data,
                                                        batch_size=1, iteration_index=j_temp, epoch_index=i_temp)
                fit_data_v[i_temp, :, 0] = np.array([temp_v])
                fit_data_u[i_temp, :, 0] = np.array([temp_u])
            for t_temp in range(params['test_epochs']):
                plt.plot(list(range(params['test_optim_iter'])), (fit_data_v[t_temp, :, :]), alpha=0.3, linewidth=2)

            ######################## visualize
            fit_data_v_df = fit_data_v.reshape((fit_data_v.shape[0],fit_data_v.shape[1]))
            fit_data_v_df = pd.DataFrame(fit_data_v_df).melt()
            ax = sns.lineplot(x="variable", y="value", data=fit_data_v_df,linewidth=7)
            ax.lines[-1].set_linestyle('-')
            plt.xlabel('iteration')
            plt.ylabel('distance between solution and ground truth (log scale)')
            if params['loss'] == 1:
                plt.title("Toy Example:" + r'$au^2-bv^2$')
            elif params['loss'] == 2:
                plt.title("Toy Example:" + r'$au^{2}-bv^{2}+2uv$')
            elif params['loss'] == 3:
                plt.title("Toy Example:" + r'$-bv \sin (a\pi u)$')


            save_name = "../../result/" + params["save_name"] 

            if not os.path.exists(save_name):
                save_name = load_name
            if not os.path.exists(save_name + "/TestingPlot/" + params['load_model']):
                os.makedirs(save_name + "/TestingPlot/" + params['load_model'])
            plt.savefig(save_name + "/TestingPlot/" + params['load_model'] + "/sample_index_" + str(j_temp) + "_v.pdf")
            plt.close()
            
            for t_temp in range(params['test_epochs']):
                plt.plot(list(range(params['test_optim_iter'])), (fit_data_u[t_temp, :, :]), alpha=0.3, linewidth=2)

            fit_data_u_df = fit_data_u.reshape((fit_data_u.shape[0],fit_data_u.shape[1]))
            fit_data_u_df = pd.DataFrame(fit_data_u_df).melt()
            ax = sns.lineplot(x="variable", y="value", data=fit_data_u_df,linewidth=7)
            ax.lines[-1].set_linestyle('-')
            
            # if params['loss'] == 3:
            #     max_scope = np.amax(fit_data_u)
            #     current_scope = 0
            #     k_temp = 0
            #     while (current_scope < max_scope):
            #         current_scope = k_temp / temp_a.cpu()
            #         plt.hlines(current_scope,xmin=0,xmax=params['test_optim_iter'])
            #         k_temp = k_temp + 1

            plt.xlabel('iteration')
            plt.ylabel('distance between solution and ground truth')
            if params['loss'] == 1:
                plt.title("Toy Example:" + r'$au^2-bv^2$')
            elif params['loss'] == 2:
                plt.title("Toy Example:" + r'$au^{2}-bv^{2}+2uv$')
            elif params['loss'] == 3:
                plt.title("Toy Example:" + r'$-bv \sin (a\pi u)$')


            save_name = "../../result/" + params["save_name"]

            if not os.path.exists(save_name):
                save_name = load_name
            if not os.path.exists(save_name + "/TestingPlot/" + params['load_model']):
                os.makedirs(save_name + "/TestingPlot/" + params['load_model'])
            plt.savefig(save_name + "/TestingPlot/" + params['load_model'] + "/sample_index_" + str(j_temp) + "_u.pdf")
            plt.close()
