import torch
import numpy as np

from metric import regression_metric


def train_test_v1(train_dataloader, test_dataloader, model, optimizer, loss_func, n_epochs, summary_writer, log_dir, logger):
    train_i = 0
    test_i = 0
    for each_epoch in range(n_epochs):
        # for each epoch, train all graphs
        true_scores = []
        for i in range(len(train_dataloader)):
            each_train_data, score = train_dataloader[i][0], train_dataloader[i][1]
            each_embedding = model(each_train_data)
            if i == 0:
                total_embeddings = each_embedding.sum()
            else:
                total_embeddings = torch.vstack((total_embeddings, each_embedding.sum()))
            true_scores.append(score)
        true_scores = torch.from_numpy(np.array(true_scores)).to(torch.float32) / 100
        predict_score = total_embeddings.squeeze()
        loss = loss_func(predict_score, true_scores)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 将结果写入我们的记录中
        logger.info('Train on epoch ' + str(each_epoch) + ' ' + str(train_i) + ' ' + str(loss.item()))
        summary_writer.add_scalar('Loss', loss.item(), train_i)
        test_i = test(each_epoch, test_dataloader, model, summary_writer, logger, test_i)
        train_i = train_i + 1


def test(each_epoch, test_dataloader, model, summary_writer, logger, test_i):
    with torch.no_grad():
        true_scores = []
        for i in range(len(test_dataloader)):
            each_train_data, score = test_dataloader[i][0], test_dataloader[i][1]
            each_embedding = model(each_train_data)
            if i == 0:
                total_embeddings = each_embedding.sum()
            else:
                total_embeddings = torch.vstack((total_embeddings, each_embedding.sum()))
            true_scores.append(score)
        true_scores = torch.from_numpy(np.array(true_scores)).to(torch.float32) / 100
        predict_score = total_embeddings.squeeze()
        metric = regression_metric(y_true=true_scores, y_pred=predict_score)
        # 将结果写入我们的记录中
        logger.info('Test on epoch ' + str(each_epoch) + ' ' + str(test_i) + ' ' + str(metric['mse']) + ' ' +
                                       str(metric['r2']) + ' ' + str(metric['r']) + ' ' + str(metric['p']))
        summary_writer.add_scalar('MSE', metric['mse'], test_i)
        summary_writer.add_scalar('R2', metric['r2'], test_i)
        summary_writer.add_scalar('r', metric['r'], test_i)
        summary_writer.add_scalar('p', metric['p'], test_i)

        test_i = test_i + 1
        return test_i