from LLM4GCL.models import BareGNN

from tqdm import tqdm
from LLM4GCL.utils import _save_checkpoint, _reload_best_model

class JointGNN(BareGNN):

    def __init__(self, task_loader, result_logger, config, checkpoint_path, dataset, model_name, seed, device):
        super(JointGNN, self).__init__(task_loader, result_logger, config, checkpoint_path, dataset, model_name, seed, device)


    def fit(self, iter):
        optimizer = self.get_optimizer(self.model)
        class_num, text_dataset, train_loader, valid_loader, test_loader = self.task_loader.get_joint_task()

        progress_bar = tqdm(range(self.config['epochs']))
        progress_bar.set_description(f'Training | Iter {iter}')

        tolerate, best_acc_valid = 0, 0.
        for epoch in range(self.config['epochs']):
            loss = self.train(epoch, self.model, text_dataset, train_loader, optimizer, class_num, self.config, self.device)
            progress_bar.write("Joint | Epoch: {} | Loss: {:.4f}".format(epoch, loss))

            if epoch > 0 and epoch % self.config['valid_epoch'] == 0:
                acc_valid, f1_valid = self.valid(self.model, text_dataset, valid_loader, class_num, self.config, self.device)
                progress_bar.write("Joint | Epoch: {} | Acc Val: {:.4f} | F1 Val: {:.4f} | Tolerate: {}".format(epoch, acc_valid, f1_valid, tolerate))
                if acc_valid > best_acc_valid:
                    tolerate = 0
                    best_acc_valid = acc_valid
                    _save_checkpoint(self.model, optimizer, epoch, self.checkpoint_path, self.dataset, self.model_name, self.seed)
                else:
                    tolerate += 1
                    if tolerate > self.config['patience']: 
                        break

            progress_bar.set_postfix({
                'Loss': f"{loss:.4f}",
                'Best Valid ACC': f"{best_acc_valid:.4f}",
                'Tolerate': tolerate
            })

            progress_bar.update(1)
        progress_bar.close()

        _reload_best_model(self.model, self.checkpoint_path, self.dataset, self.model_name, self.seed)
        curr_acc_test, curr_f1_test = self.evaluate(self.model, text_dataset, test_loader, class_num, self.config, self.device)

        print("Joint | Acc Test: {:.4f} | F1 Test: {:.4f}".format(curr_acc_test, curr_f1_test))

        for curr_session in range(self.session_num):
            acc_list = []
            for s in range(curr_session + 1):
                _, text_dataset_iso, _, _, _, test_loader_isolate, _ = self.task_loader.get_task(s)
                prev_acc_test_isolate, prev_f1_test_isolate = self.evaluate(self.model, text_dataset_iso, test_loader_isolate, class_num, self.config, self.device)
                acc_list.append(prev_acc_test_isolate)

            _, _, text_dataset_joint, _, _, _, test_loader_joint = self.task_loader.get_task(curr_session)
            curr_acc_test_joint, curr_f1_test_joint = self.evaluate(self.model, text_dataset_joint, test_loader_joint, class_num, self.config, self.device)

            self.result_logger.add_new_results(acc_list, curr_acc_test_joint)

        return self.result_logger