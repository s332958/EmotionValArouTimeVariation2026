import utils

PATH_DATASET = "FabioCode/TRAIN_RELEASE_3SEP2025/train_subtask1.csv"

TRAINING = True

MODEL_NAME = None
PATH_SAVE_MODEL = "FabioCode/models"
DEVICE = "cpu"
BATCH_SIZE = 32
EPOCHS = 10
LR = 0.01
NEW_MODEL_NAME = None


if __name__ == "__main__":

    dataset = utils.extract_data(PATH_DATASET,number_class=5,offset=2)

    if TRAINING:
        model = utils.train_model(dataset,MODEL_NAME,PATH_SAVE_MODEL,BATCH_SIZE,EPOCHS,LR,DEVICE,NEW_MODEL_NAME)

