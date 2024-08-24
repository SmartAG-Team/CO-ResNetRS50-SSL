# import numpy as np

# def calculate_metrics(conf_matrix):
#     # True Positives, False Positives, False Negatives, True Negatives
#     TP = np.diag(conf_matrix)
#     FP = np.sum(conf_matrix, axis=0) - TP
#     FN = np.sum(conf_matrix, axis=1) - TP
#     TN = np.sum(conf_matrix) - (FP + FN + TP)

#     # Calculate metrics
#     accuracy = np.sum(TP) / np.sum(conf_matrix)
#     recall = TP / (TP + FN)
#     precision = TP / (TP + FP)
#     f1_score = 2 * (precision * recall) / (precision + recall)

#     # Handle NaN values which occur when precision + recall = 0
#     recall = np.nan_to_num(recall)
#     precision = np.nan_to_num(precision)
#     f1_score = np.nan_to_num(f1_score)

#     avg_recall = np.mean(recall)
#     avg_precision = np.mean(precision)
#     avg_f1_score = np.mean(f1_score)

#     return accuracy, avg_recall, avg_precision, avg_f1_score

# # Confusion matrices
# ConvNext = np.array([[473, 16, 0, 0, 0, 0, 0],
#                      [19, 467, 33, 4, 1, 2, 3],
#                      [0, 42, 349, 78, 18, 7, 3],
#                      [0, 1, 57, 575, 19, 4, 1],
#                      [0, 0, 13, 53, 159, 18, 1],
#                      [1, 2, 2, 5, 4, 290, 20],
#                      [4, 3, 0, 2, 3, 14, 477]])

# ShuffleNet = np.array([[476, 13, 0, 0, 0, 0, 0],
#                        [19, 475, 28, 5, 0, 1, 1],
#                        [0, 41, 401, 48, 4, 3, 0],
#                        [0, 3, 39, 598, 13, 3, 1],
#                        [0, 0, 9, 31, 196, 8, 0],
#                        [0, 0, 3, 1, 5, 298, 17],
#                        [0, 0, 1, 5, 0, 12, 485]])

# EfficientNet = np.array([[475, 11, 1, 1, 0, 0, 1],
#                          [17, 481, 25, 3, 0, 1, 2],
#                          [0, 41, 383, 61, 9, 2, 1],
#                          [0, 1, 46, 594, 8, 4, 4],
#                          [0, 0, 5, 17, 216, 6, 0],
#                          [0, 0, 2, 2, 5, 301, 14],
#                          [0, 0, 0, 5, 0, 16, 482]])

# AlexNet = np.array([[473, 12, 0, 0, 0, 0, 4],
#                     [21, 461, 39, 4, 1, 0, 3],
#                     [0, 37, 372, 61, 20, 5, 2],
#                     [0, 0, 64, 551, 34, 4, 4],
#                     [0, 0, 16, 62, 156, 10, 0],
#                     [0, 0, 4, 5, 9, 289, 17],
#                     [0, 2, 0, 5, 0, 21, 475]])

# VGG = np.array([[470, 16, 0, 1, 0, 0, 2],
#                 [25, 453, 39, 5, 1, 2, 4],
#                 [0, 46, 357, 68, 14, 6, 6],
#                 [0, 2, 68, 560, 15, 5, 7],
#                 [0, 1, 21, 36, 171, 14, 1],
#                 [1, 2, 6, 5, 15, 277, 18],
#                 [1, 1, 4, 6, 1, 20, 470]])

# CO_ResNetRS50 = np.array([[484, 5, 0, 0, 0, 0, 0],
#                           [19, 484, 24, 2, 0, 0, 0],
#                           [0, 28, 417, 43, 7, 2, 0],
#                           [0, 0, 37, 612, 3, 4, 1],
#                           [0, 0, 6, 19, 212, 7, 0],
#                           [0, 2, 1, 1, 3, 306, 11],
#                           [0, 0, 0, 3, 1, 11, 488]])

# # Models list
# models = [ConvNext, ShuffleNet, EfficientNet, AlexNet, VGG, CO_ResNetRS50]
# model_names = ["ConvNext", "ShuffleNet", "EfficientNet", "AlexNet", "VGG", "CO_ResNetRS50"]

# # Calculate metrics for each model
# results = {}
# for name, conf_matrix in zip(model_names, models):
#     accuracy, avg_recall, avg_precision, avg_f1_score = calculate_metrics(conf_matrix)
#     results[name] = {
#         "Accuracy": accuracy,
#         "Recall": avg_recall,
#         "Precision": avg_precision,
#         "F1-score": avg_f1_score
#     }

# print(results)



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
ResNetRS50 = np.array([[479, 8, 1, 0, 0, 0, 1],
                    [15, 471, 35, 5, 0, 0, 3],
                    [0, 35, 404, 51, 5, 1, 1],
                    [0, 1, 41, 606, 7, 1, 1],
                    [0, 0, 6, 21 , 207, 9, 1],
                    [0, 0, 1, 1, 6, 303, 13],
                    [0, 1, 1, 3, 1, 20, 477]])

ResNetRS50_SSL = np.array([[481, 7, 0, 0, 0, 0, 1],
                       [20, 473, 32, 1, 0, 1, 2],
                       [0, 33, 399, 59, 4, 2, 0],
                       [0, 1, 47, 596, 10, 2, 1],
                       [0, 0, 6, 17, 219, 1, 1],
                       [0, 0, 1, 2, 11, 300, 10],
                       [0, 0, 0, 5, 0, 9, 489]])

C_ResNetRS50_SSL = np.array([[480, 9, 0, 0, 0, 0, 0],
                         [17, 480, 28, 2, 0, 0, 2],
                         [0, 37, 412, 44, 3, 1, 0],
                         [0, 3, 40, 603, 8, 0, 3],
                         [0, 0, 0, 24, 211, 9, 0],
                         [0, 0, 2, 0, 2, 307, 13],
                         [0, 0, 0, 4, 0, 7, 492]])

CO_ResNetRS50_SSL = np.array([[484, 5, 0, 0, 0, 0, 0],
                          [19, 484, 24, 2, 0, 0, 0],
                          [0, 28, 417, 43, 7, 2, 0],
                          [0, 0, 37, 612, 3, 4, 1],
                          [0, 0, 6, 19, 212, 7, 0],
                          [0, 2, 1, 1, 3, 306, 11],
                          [0, 0, 0, 3, 1, 11, 488]])

# Models list
models = [ResNetRS50, ResNetRS50_SSL, C_ResNetRS50_SSL, CO_ResNetRS50_SSL]
model_names = ["ResNetRS50", "ResNetRS50_SSL", "C_ResNetRS50_SSL", "CO_ResNetRS50_SSL"]

# Calculate metrics for each model
results = {}
for name, conf_matrix in zip(model_names, models):
    accuracy, avg_recall, avg_precision, avg_f1_score = calculate_metrics(conf_matrix)
    results[name] = {
        "Accuracy": accuracy,
        "Recall": avg_recall,
        "Precision": avg_precision,
        "F1-score": avg_f1_score
    }

print(results)