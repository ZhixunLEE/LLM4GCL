import numpy as np

class CLMetric():

    def __init__(self, ):
        self.task_matrix = [] # for AA / AF
        self.task_list = [] # for AA / LA

    def add_new_results(self, acc_list, curr_acc):
        self.task_matrix.append(acc_list)
        self.task_list.append(curr_acc)

    def get_results(self, ):
        # AA / AF
        task_num = len(self.task_matrix)
        for task_id, task in enumerate(self.task_matrix):
            print(f"Task {task_id + 1}: " + " ".join([f"{x:.4f}" for x in task]))
        final_acc = np.array(self.task_matrix[-1])

        final_fgt = []
        for i in range(task_num - 1):
            final_fgt.append(self.task_matrix[task_num - 1][i] - self.task_matrix[i][i])
        final_fgt = np.array(final_fgt)

        avg_acc_iso = np.mean(final_acc)
        avg_fgt_iso = np.mean(final_fgt)

        # AA / LA
        for i, acc in enumerate(self.task_list):
            print(f"Task {i + 1} | ACC: {acc:.4f}")

        all_task_acc = np.array(self.task_list)
        avg_acc_jot = np.mean(all_task_acc)
        last_acc_jot = all_task_acc[-1]

        return avg_acc_iso, avg_fgt_iso, avg_acc_jot, last_acc_jot


def compute_metrics(p):
    from sklearn.metrics import accuracy_score
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    return {"accuracy": accuracy}
