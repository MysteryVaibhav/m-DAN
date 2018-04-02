from util import *


class Evaluator:
    def __init__(self, params, data_loader):
        self.params = params
        self.data_loader = data_loader

    def recall(self, model, is_test=False):
        """
        :param model:
        :param is_test:
        :return: Returns the recall at 1,5,10
        """
        test_z_v = None  # no_of_images * hidden_dimension
        test_z_w = None  # no_of_images * hidden_dimension
        if is_test:
            ids = self.data_loader.test_ids
            plain_ids = self.data_loader.plain_test_ids
            data_loader_1 = self.data_loader.test_data_loader_1
            data_loader_2 = self.data_loader.test_data_loader_2
        else:
            ids = self.data_loader.val_ids
            plain_ids = self.data_loader.plain_val_ids
            data_loader_1 = self.data_loader.eval_data_loader_1
            data_loader_2 = self.data_loader.eval_data_loader_2

        for (caption_, mask_, image_, concept_) in data_loader_1:
            _, _, z_v, z_w = model(to_variable(caption_),
                              to_variable(mask_),
                              to_variable(image_),
                              to_variable(concept_),
                              True)
            if test_z_v is None:
                test_z_v = z_v.data.cpu()
                test_z_w = z_w.data.cpu()
            else:
                test_z_v = torch.cat((test_z_v, z_v.data.cpu()), 0)
                test_z_w = torch.cat((test_z_w, z_w.data.cpu()), 0)
        test_z_v = test_z_v.numpy()
        test_z_w = test_z_w.numpy()

        r_1 = 0
        r_5 = 0
        r_10 = 0
        for (caption_, mask_, image_, label, concept_) in data_loader_2:
            _, z_u, _, _ = model(to_variable(caption_),
                              to_variable(mask_),
                              to_variable(image_),
                              to_variable(concept_),
                              True)
            z_u = z_u.data.cpu().numpy()

            # Compute similarity with the existing images
            similarity = np.matmul(test_z_v * test_z_w, z_u.T)
            for column in range(similarity.shape[1]):
                top_10_img_idx = (-similarity[:, column]).argsort()[:10]
                if label[column] == plain_ids[top_10_img_idx[0]]:
                    r_1 += 1
                    r_5 += 1
                    r_10 += 1
                elif label[column] in [plain_ids[x] for x in top_10_img_idx[1:5]]:
                    r_5 += 1
                    r_10 += 1
                elif label[column] in [plain_ids[x] for x in top_10_img_idx[6:10]]:
                    r_10 += 1

        return r_1 / len(ids), r_5 / len(ids), r_10 / len(ids)
