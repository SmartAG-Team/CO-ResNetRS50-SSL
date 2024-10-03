import numpy as np

def calculate_metrics(conf_matrix):
    # True Positives, False Positives, False Negatives, True Negatives
    TP = np.diag(conf_matrix)
    FP = np.sum(conf_matrix, axis=0) - TP
    FN = np.sum(conf_matrix, axis=1) - TP
    TN = np.sum(conf_matrix) - (FP + FN + TP)

    # Calculate metrics
    accuracy = np.sum(TP) / np.sum(conf_matrix)
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    f1_score = 2 * (precision * recall) / (precision + recall)

    # Handle NaN values which occur when precision + recall = 0
    recall = np.nan_to_num(recall)
    precision = np.nan_to_num(precision)
    f1_score = np.nan_to_num(f1_score)

    avg_recall = np.mean(recall)
    avg_precision = np.mean(precision)
    avg_f1_score = np.mean(f1_score)

    return accuracy, avg_recall, avg_precision, avg_f1_score

# Confusion matrices
ResNetRS50 = np.array([[330, 12, 1, 0, 0, 1, 1],
                     [17, 337, 28, 3, 0, 1, 3],
                     [1, 16, 265, 38, 4, 0, 0],
                     [0, 2, 46, 396, 10, 0, 3],
                     [0, 1, 8, 16, 139, 6, 2],
                     [0, 1, 3, 1, 6, 222, 6],
                     [1, 0, 4, 3, 1, 13, 322]])

ResNetRS50_Aug = np.array([[331, 12, 0, 1, 0, 1, 0],
                        [11, 355, 22, 1, 0, 0, 0],
                        [0, 26, 258, 38, 2, 0, 0],
                        [0, 2, 40, 404, 9, 0, 2],
                        [0, 2, 7, 24, 133, 5, 1],
                        [1, 2, 3, 1, 5, 218, 9],
                        [0, 1, 4, 6, 1, 12, 320]])


ResNetRS50_SSL_85 = np.array([[331, 13, 0, 1, 0, 0, 0],
                     [16, 356, 15, 2, 0, 0, 0],
                     [0, 20, 254, 45, 5, 0, 0],
                     [0, 3, 30, 410, 9, 1, 4],
                     [0, 0, 6, 24, 137, 4, 1],
                     [0, 0, 4, 3, 8, 216, 8],
                     [1, 3, 3, 4, 1, 9, 323]])


ResNetRS50_SSL_875 = np.array([[328, 15, 1, 0, 0, 1, 0],
                     [16, 352, 18, 2, 0, 0, 1],
                     [0, 22, 255, 43, 3, 1, 0],
                     [0, 3, 43, 404, 4, 1, 2],
                     [0, 1, 11, 29, 127, 4, 0],
                     [0, 2, 3, 1, 6, 217, 10],
                     [1, 3, 1, 7, 1, 10, 321]])


ResNetRS50_SSL_90 = np.array([[328, 14, 1, 0, 0, 1, 1],
                     [14, 350, 19, 1, 0, 1, 4],
                     [0, 12, 259, 50, 3, 0, 0],
                     [0, 3, 34, 410, 7, 0, 3],
                     [0, 2, 6, 20, 137, 7, 0],
                     [0, 1, 2, 1, 3, 222, 10],
                     [0, 1, 4, 2, 1, 13, 323]])


ResNetRS50_SSL_925 = np.array([[327, 16, 1, 0, 0, 1, 0],
                     [13, 346, 25, 1, 0, 1, 3],
                     [1, 16, 264, 40, 2, 1, 0],
                     [0, 1, 24, 424, 6, 0, 2],
                     [0, 0, 5, 26, 136, 5, 0],
                     [0, 1, 3, 1, 4, 220, 10],
                     [0, 13, 0, 8, 0, 12, 311]])


ResNetRS50_SSL_95 = np.array([[330, 14, 0, 1, 0, 0, 0],
                     [13, 352, 22, 1, 0, 0, 1],
                     [0, 20, 255, 47, 2, 0, 0],
                     [0, 2, 32, 411, 9, 1, 2],
                     [0, 1, 8, 21, 136, 5, 1],
                     [1, 1, 5, 2, 8, 213, 9],
                     [1, 3, 5, 2, 2, 5, 326]])

ResNetRS50_SSL_LabelAug = np.array([[331, 12, 0, 1, 0, 1, 0],
                        [13, 351, 23, 2, 0, 0, 0],
                        [0, 19, 261, 40, 4, 0, 0],
                        [0, 2, 32, 410, 9, 2, 2],
                        [0, 0, 8, 31, 126, 6, 1],
                        [0, 1, 3, 2, 7, 215, 11],
                        [0, 0, 3, 4, 0, 8, 329]])

ResNetRS50_SSL_AllAug = np.array([[330, 14, 1, 0, 0, 0, 0],
                        [18, 341, 25, 1, 0, 1, 3],
                        [1, 17, 247, 55, 4, 0, 0],
                        [0, 5, 21, 418, 10, 1, 2],
                        [0, 0, 6, 16, 146, 3, 1],
                        [0, 2, 1, 1, 6, 220, 9],
                        [0, 4, 1, 4, 1, 9, 325]])


C_ResNetRS50_SSL = np.array([[330, 13, 0, 1, 0, 0, 1],
                         [13, 356, 20, 0, 0, 0, 0],
                         [0, 18, 253, 47, 4, 2, 0],
                         [0, 1, 31, 420, 3, 0, 2],
                         [0, 2, 6, 20, 140, 3, 1],
                         [0, 0, 2, 3, 3, 223, 8],
                         [0, 2, 1, 3, 2, 17, 319]])


CO_ResNetRS50_SSL = np.array([[333, 10, 1, 0, 0, 0, 1],
                          [19, 344, 24, 0, 0, 1, 1],
                          [0, 17, 262, 39, 6, 0, 0],
                          [0, 2, 27, 420, 5, 1, 2],
                          [0, 0, 7, 16, 145, 3, 1],
                          [0, 1, 4, 1, 3, 222, 8],
                          [0, 1, 3, 3, 0, 13, 324]])



# Models list
models = [ResNetRS50, ResNetRS50_Aug, ResNetRS50_SSL_85, ResNetRS50_SSL_875, ResNetRS50_SSL_90, ResNetRS50_SSL_925, ResNetRS50_SSL_95,
            ResNetRS50_SSL_LabelAug, ResNetRS50_SSL_AllAug, C_ResNetRS50_SSL, CO_ResNetRS50_SSL]
model_names = ["ResNetRS50", "ResNetRS50-Aug", "ResNetRS50-SSL-85", "ResNetRS50-SSL-875", "ResNetRS50-SSL-90", "ResNetRS50-SSL-925", "ResNetRS50-SSL-95", 
               "ResNetRS50-SSL-LabelAug", "ResNetRS50-SSL-AllAug", "C-ResNetRS50-SSL", "CO-ResNetRS50-SSL"]

# Calculate metrics for each model
with open("D:/Desktop/metrics_results.txt", "w") as file:
    for name, conf_matrix in zip(model_names, models):
        accuracy, avg_recall, avg_precision, avg_f1_score = calculate_metrics(conf_matrix)
        file.write(f"Model: {name}\n")
        file.write(f"Accuracy: {accuracy:.4f}\n")
        file.write(f"Recall: {avg_recall:.4f}\n")
        file.write(f"Precision: {avg_precision:.4f}\n")
        file.write(f"F1-score: {avg_f1_score:.4f}\n")
        file.write("\n" + "-"*40 + "\n\n")
