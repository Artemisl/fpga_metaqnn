import math
import numpy as np
import os
import sys

from string_to_model import Parser

class ModelGen:

    def __init__(self, model_dir, hyper_parameters, state_space_parameters):
        self.model_dir = model_dir
        self.ssp = state_space_parameters
        self.hp = hyper_parameters

    def model_paths(self, model_descr):
        solver_path = os.path.join(self.model_dir, "solver.prototxt")
        netspec_path = os.path.join(self.model_dir, "train_net.prototxt")
        return self.model_dir, solver_path, netspec_path

    def compile_openvino_bin_file(self, copy_caffemodel_path):
        caffe_model_optimizer_path ='/opt/intel/2019_r1/openvino/deployment_tools/model_optimizer/mo_caffe.py'
        log_path = os.path.join(self.model_dir, "compile_openvino_bin_file.log")
        caffe_input_model = self.model_dir + ''
        print '[1]Run Command : python3.5 %s --input_model %s --output_dir %s --data_type FP16 >> %s' % (caffe_model_optimizer_path, copy_caffemodel_path, self.model_dir, log_path)
        os.system('python3.5 %s --input_model %s --output_dir %s --data_type FP16 >> %s' % (caffe_model_optimizer_path, copy_caffemodel_path, self.model_dir, log_path))

    def inference_caffemodel_in_fpga(self):
        openvino_bin_path = '/root/inference_engine_samples_build/intel64/Release/classification_sample_artemis'
        cifar10_picture_path = '/opt/intel/2019_r1/openvino/deployment_tools/terasic_demo/demo/pic_video/horse5.png'
        log_path = os.path.join(self.model_dir, "inference_caffemodel_in_fpga.log")
        caffemodel_openvino_xml_path = os.path.join(self.model_dir, "train_net_mo.xml")
        print '[2]Run Command : %s -d HETERO:FPGA,CPU -i %s -m %s >> %s' % (openvino_bin_path, cifar10_picture_path, caffemodel_openvino_xml_path, log_path)
        os.system('%s -d HETERO:FPGA,CPU -i %s -m %s >> %s' % (openvino_bin_path, cifar10_picture_path, caffemodel_openvino_xml_path, log_path))

    # Saves caffe model for model optimizer used in fpga.
    def save_models_for_mo(self, model_descr):
        print "Saving Models for Model Optimizer"
        model_dir, solver_path, netspec_path = self.model_paths(model_descr)
        Parser(self.hp, self.ssp).create_caffe_spec_for_model_optimizer(model_dir, netspec_path)

        caffemodel_path = os.path.join(self.model_dir, "modelsave_iter_1.caffemodel")
        copy_caffemodel_path = os.path.join(self.model_dir, "train_net_mo.caffemodel")
        copy_caffemodel_command = 'cp %s %s' % (caffemodel_path, copy_caffemodel_path)
        print "copy_caffemodel_command : %s" % copy_caffemodel_command
        os.system(copy_caffemodel_command)

        cifar10_label_ori_path = "/root/metaqnn/train_net_mo.labels"
        cifar10_label_dist_path = os.path.join(self.model_dir, "train_net_mo.labels")
        copy_cifar10_label_command = 'cp %s %s' % (cifar10_label_ori_path, cifar10_label_dist_path)
        print "copy_cifar10_label_command : %s" % copy_cifar10_label_command
        os.system(copy_cifar10_label_command)

        ori_compile_log_path = os.path.join(self.model_dir, "compile_openvino_bin_file.log")
        ori_inference_log_path = os.path.join(self.model_dir, "inference_caffemodel_in_fpga.log")
        os.system('rm %s' % ori_compile_log_path)
        os.system('rm %s' % ori_inference_log_path)

        self.compile_openvino_bin_file(copy_caffemodel_path)
        for i in range(self.hp.INFERENCE_TIMES):
            self.inference_caffemodel_in_fpga()

    # Saves caffe specs including solver to given directories.
    def save_models(self, model_descr, learning_rate, max_iter):
        print "Creating Caffe Configs for %s" % model_descr
        model_dir, solver_path, netspec_path = self.model_paths(model_descr)
        p = Parser(self.hp, self.ssp)
        p.create_caffe_spec(model_descr, netspec_path)
        self.save_solver(solver_path, netspec_path, learning_rate, max_iter)
        return model_dir, solver_path

    # Saves the solver from hyper parameters.
    def save_solver(self, solver_path, netspec_path, learning_rate=-1, max_iter=-1):
        if learning_rate == -1:
            learning_rate = self.hp.INITIAL_LEARNING_RATES[0]
        if max_iter == -1:
            max_iter = self.hp.MAX_STEPS
        solver_proto =  'net: "%s"' % netspec_path + \
                        '\ntest_iter: %d' % (self.hp.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL/self.hp.EVAL_BATCH_SIZE) + \
                        '\ntest_interval: %d' % (self.hp.TEST_INTERVAL_EPOCHS*self.hp.NUM_ITER_PER_EPOCH_TRAIN,) + \
                        '\nbase_lr: %f' % learning_rate + \
                        '\nmomentum: %f' % self.hp.MOMENTUM + \
                        '\nweight_decay: %f' % self.hp.WEIGHT_DECAY_RATE + \
                        '\ndisplay: %d' % self.hp.DISPLAY_ITER + \
                        '\nmax_iter: %d' % max_iter + \
                        '\nsnapshot: %d' % (self.hp.SAVE_EPOCHS*self.hp.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN/self.hp.TRAIN_BATCH_SIZE) + \
                        '\nsnapshot_prefix: "%s/%s"' % (self.model_dir, 'modelsave') + \
                        '\nsolver_mode: %s' % 'GPU' + \
                        '\ntype: "%s"' % self.hp.OPTIMIZER + \
                        '\nlr_policy: "%s"' % self.hp.LR_POLICY + \
                        '\ngamma: %f' % self.hp.LEARNING_RATE_DECAY_FACTOR

        if max_iter == 1:
            solver_proto =  'net: "%s"' % netspec_path + \
                        '\ntest_iter: %d' % (1) + \
                        '\ntest_interval: %d' % (1) + \
                        '\nbase_lr: %f' % learning_rate + \
                        '\nmomentum: %f' % self.hp.MOMENTUM + \
                        '\nweight_decay: %f' % self.hp.WEIGHT_DECAY_RATE + \
                        '\ndisplay: %d' % self.hp.DISPLAY_ITER + \
                        '\nmax_iter: %d' % max_iter + \
                        '\nsnapshot: %d' % (self.hp.SAVE_EPOCHS*self.hp.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN/self.hp.TRAIN_BATCH_SIZE) + \
                        '\nsnapshot_prefix: "%s/%s"' % (self.model_dir, 'modelsave') + \
                        '\nsolver_mode: %s' % 'GPU' + \
                        '\ntype: "%s"' % self.hp.OPTIMIZER + \
                        '\nlr_policy: "%s"' % self.hp.LR_POLICY + \
                        '\ngamma: %f' % self.hp.LEARNING_RATE_DECAY_FACTOR
        if self.hp.LR_POLICY == 'step':
            solver_proto += '\nstepsize: %i' % (self.hp.NUM_EPOCHS_PER_DECAY *self.hp.NUM_ITER_PER_EPOCH_TRAIN)

        else:
            for epoch, number_decays in self.hp.STEP_LIST:
                step = epoch * self.hp.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / self.hp.TRAIN_BATCH_SIZE
                for j in range(number_decays):
                    solver_proto += '\nstepvalue: %i' % (step + j)
            
        with open(solver_path, "w") as solver:
            solver.write(solver_proto)
