import torch
import tqdm
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from data_handle.code_enums import mode_input_data
import time
from torch import jit
class eval_proc():
    def __init__(self, devset_gen,device,w_fp,save_dir):
        # self.model  = model
        self.data = devset_gen
        self.device =device
        self.best_eer=99.
        self.best_high_er=99.
        self.fp_t = w_fp
        # self.test= test_proc
        self.save_dir= save_dir
        self.f_eer =open(save_dir + 'eers.txt', 'a', buffering=1)

        return

    def eval_process(self,model,tqdm,test_proc,epoch,use_feer,parser,experiment,str_f="_reg_"):
        model.eval()
        self.f_eer.write('%d ' % epoch)
        with torch.set_grad_enabled(False):
            with tqdm(total=len(self.data), ncols=70) as pbar:

                y_score = []  # score for each sample
                y = []  # label for each sample
                cntr2 = 0

                for m_batch, m_label in self.data:

                    cntr2 += 1
                    y.extend(list(m_label))
                    # start=time.time()
                    _,out= model(m_batch)
                    # print ("sec ",time.time()-start)
                    test_proc.measure_res(out, m_label)
                    print("Inter Output eval ", test_proc.c0_m0, test_proc.c0_m1, test_proc.c1_m1, test_proc.c1_m0)
                    y_score.extend(out.cpu().numpy()[:, 0])  # >>> (16, 64?)
                    pbar.update(1)
            if use_feer:
                self.estimate_eer(model,epoch,y_score,y,test_proc,parser,experiment,str_f)
            else:
                print("Output eval ",test_proc.c0_m0, test_proc.c0_m1, test_proc.c1_m1, test_proc.c1_m0)
                test_proc.c0_m0=0
                test_proc.c0_m1=0
                test_proc.c1_m1=0
                test_proc.c1_m0=0

    # https: // gist.github.com / aqzlpm11 / 9e33a20c5e8347537bec532ae7319ba8
    def estimate_eer(self, model, epoch, y_score, y, test_proc,   parser, experiment,str_f):
            f_res = open(self.save_dir + 'results/epoch%s.txt' % (epoch), 'w')
            for _s, _t in zip(y, y_score):
               f_res.write('{score} {target}\n'.format(score=_s, target=_t))
                # print ("ltt")
            f_res.close()
            print("Output eval ", test_proc.c0_m0, test_proc.c0_m1, test_proc.c1_m1, test_proc.c1_m0)
            fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=0)
            kinter= interp1d(fpr, tpr)
            # eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
            eer = brentq(lambda x: 1. - x - kinter(x), 0., 1.)
            high_s= 1-kinter(self.fp_t)
            print(eer)
            print ("hig dim", high_s)
            if not bool(parser['comet_disable']):
                    experiment.log_metric('val_eer', eer)
            self.f_eer.write('%f \n' % eer)

            # record best validation model
            if float(high_s) < self.best_high_er:
                self.best_high_er=high_s
                print('New best high er: %f' % float(high_s))
                dir_best_model_weights = self.save_dir + 'models/%d-%.6f-and-%.6f.h5' % (epoch, high_s,eer)
                if len(parser['gpu_idx']) > 1:  # multi GPUs
                        torch.save(model.module.state_dict(), self.save_dir + 'models/'+str_f+'best.pth')
                else:  # single GPU
                        torch.save(model.state_dict(), self.save_dir + 'models/'+str_f+'best.pth')

            if float(eer) < self.best_eer:
                    print('New best EER: %f' % float(eer))
                    self.best_eer= float(eer)
                    # dir_best_model_weights = self.save_dir + 'models/%d-%.6f.h5' % (epoch, eer)
                    # if not bool(parser['comet_disable']):
                    #     experiment.log_metric('best_val_eer', eer)
                    #
                    # # save best model
                    # if len(parser['gpu_idx']) > 1:  # multi GPUs
                    #     torch.save(model.module.state_dict(), self.save_dir + 'models/best.pt')
                    # else:  # single GPU
                    #     torch.save(model.state_dict(), self.save_dir + 'models/best.pt')

            if not bool(parser['save_best_only']):
                    # save model
                    if len(parser['gpu_idx']) > 1:  # multi GPUs
                        torch.save(model.module.state_dict(), self.save_dir + 'models/%d-%.6f-%.6f.pth' % (epoch, high_s,eer))
                    else:  # single GPU
                        torch.save(model.state_dict(), self.save_dir + 'models/%d-%.6f-%.6f.pth' % (epoch,high_s, eer))
