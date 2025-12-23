import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaModel, RobertaTokenizer
import pandas as pd
import time
import os


def create_class_per_loss(values: list, number_class: int, offset: int):
    """
    Funzione per creare un vettore di 0 di dim num_classi, 
    lo scopo e' quello di poter applicare una cross-entropy,
    ovvero il modello dovrebbe predirre per ogni classe la probabita di essere in quella classe.
    Percio converto le label in vettori di dim pari al numero di classi, saranno tutti 0 tranne la classe corretta a 1
    """
    values_torch = torch.tensor(values, dtype=torch.long)
    labels_mapped = values_torch + offset  
    one_hot = torch.zeros((len(values), number_class), dtype=torch.float)
    one_hot[torch.arange(len(values)), labels_mapped] = 1.0
    return one_hot


def extract_data(path_file, number_class=5, offset=2):
    """
    Funzione per estrarre le colonne utili dal dataset,
    in questo caso (Task 1) si prende il testo, is_words, valence, arousal
    is_words e' usato per dividere i testi in sequenze di parole
    il risultato e' un dataframe pd con:
    testo:str, valence:torch.tensor(float, dim=num_class), arousal:float
    """
    cols_to_read = ["text", "is_words", "valence", "arousal"]
    df = pd.read_csv(path_file, usecols=cols_to_read)

    data = []
    for row in df.itertuples(index=False):
        text = row.text
        is_words = row.is_words
        valence = row.valence
        arousal = row.arousal

        valence = create_class_per_loss([valence],number_class,offset).squeeze(0)

        if is_words:
            words = text.split(",")
            for w in words:
                data.append(
                    {
                        "text" : str(w),
                        "valence": valence,
                        "arousal" : float(arousal)
                    }
                )

        else:
            data.append(
                {
                    "text" : str(text),
                    "valence": valence,
                    "arousal" : float(arousal)
                }
            )

    return pd.DataFrame(data)



### MODEL ###

class Model(nn.Module):

    """
    Modello di predizione basato su RoBERTa, 
    come input prende il tipo di RoBERTa, e il numero di classi di output.
    E' composto da un tokenizer per RoBERTa, RoBERTa,
    e due teste per predirre la classe di valence e il valore di arousal
    (Attualmente uso dei MLP singoli possono essere cambiati per ottenere relazioni piu complesse)
    (Siccome si fa training con cross-entropy non metto una softmax alla fine per fare la classificazione nella parte di valence)
    """

    def __init__(self,
                 RobertaType = "roberta-base",
                 outputsClass = 5,
                 ):
        
        super().__init__()

        # tokenizer e RoBERTa
        self.LAYER_tokenizer = RobertaTokenizer.from_pretrained(RobertaType)
        self.LAYER_RoBERTa = RobertaModel.from_pretrained(RobertaType)

        # teste per la classificazione e regressione
        self.LAYER_headValcenceSoftmax = nn.Softmax(outputsClass)
        self.LAYER_headValence = nn.Sequential(
            # Aggiungere qui nuovi layer per ottenere maggiore capacita espressiva
            nn.Linear(self.LAYER_RoBERTa.config.hidden_size, outputsClass)
            )
        self.LAYER_headArousal = nn.Sequential(
            # Aggiungere qui nuovi layer per ottenere maggiore capacita espressiva
            nn.Linear(self.LAYER_RoBERTa.config.hidden_size, 1)
            )


    def forward(self, phrases, device):

        # il tokenizer mi restituisce i tensori delle parti del testo, restiuisce il valore del token e la maschera (token vero 1, fittizio 0)
        # si applica il padding per ottenere tutte le frasi con la stessa misura
        token_struct = self.LAYER_tokenizer(phrases, return_tensors="pt", padding=True, truncation=True)
        # muovo i valori del tokenizer sul device altrimenti da errore
        token_struct = {k: v.to(device) for k, v in token_struct.items()}
        roberta_struct =  self.LAYER_RoBERTa(**token_struct)

        # prendo output dell'ultimo hidden state
        hidden_state = roberta_struct.last_hidden_state
        # pooler output molto utile perche genera un valore a tutta la frase ma da addestrare (da capire come fare)
        pooler_output = roberta_struct.pooler_output

        # tramite le teste ottengo la classificazione e regressione
        valence = self.LAYER_headValence(hidden_state[:, 0, :])
        valence = self.LAYER_headValcenceSoftmax(valence)
        arousal = self.LAYER_headArousal(hidden_state[:, 0, :])

        return valence, arousal
    

    def forward_heads_only(self, phrases, device):
        """ 
        Questa funzione si puo usare nel training per accelerare il processo,
        di fatto si allenano solo le teste e i valori di RoBERTa non vengono modificati
        (Capire se puo essere una soluzione)
        """
        token_struct = self.LAYER_tokenizer(phrases, return_tensors="pt", padding=True, truncation=True)
        token_struct = {k: v.to(device) for k, v in token_struct.items()}

        with torch.no_grad():  # blocca gradiente su RoBERTa
            hidden_state = self.LAYER_RoBERTa(**token_struct).last_hidden_state

        cls_token = hidden_state[:, 0, :]
        valence = self.LAYER_headValence(cls_token)
        arousal = self.LAYER_headArousal(cls_token)
        return valence, arousal


### DATASET PERSONALIZZATO ###

class CustomDataset(Dataset):
    def __init__(self, df):
        """
        df: DataFrame con colonne 'text', 'valence', 'arousal'
        """
        self.texts = df["text"].tolist()
        self.valence = torch.stack(df["valence"].tolist())  # già one-hot
        self.arousal = torch.tensor(df["arousal"].tolist(), dtype=torch.float)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.valence[idx], self.arousal[idx]
    

### FUNZIONE DI TRAINING ###

def load_model(path_model, device):
    model = Model()
    model.load_state_dict(torch.load(path_model, map_location=device))
    return model

def train_model(df, model_name=None, path_save="", batch_size=8, epochs=3, lr=2e-5, device='cpu', name_model_save = None):
    
    """
    df: DataFrame del dataset
    model: name of the model to load if is None create new model 
    batch_size: batch dim 
    epochs: epochs number
    lr: learning rate
    device: 'cpu' or 'cuda'


    La funzione serve per allenare nuovi modelli partendo da uno specifico,
    se non indicato si crea un nuovo modello con solo RoBERTa allenato parzialmente.
    ATTENZIONE: Se il nome del modello di output risulta gia esistente si carica tale modello 
                e non si effettua il training


    """

    model = Model()

    if model_name is not None:
        model = load_model(f"{path_save}/{model_name}",device)

    model.to(device)
    model.train()

    if model_name is None:
        model_name = "Model.pth"

    if name_model_save is None:
        t = model_name.strip(".pth")
        name_model_save = f"{t}_trained.pth"

    if os.path.exists(f"{path_save}/{name_model_save}"):
        print("Model already trained or existent, please select new name!")
        model = load_model(f"{path_save}/{name_model_save}",device)
        model.eval()
        return model

    # Dataset e dataloader
    dataset = CustomDataset(df)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Loss functions
    loss_valence = nn.CrossEntropyLoss()   # per classificazione valence
    loss_arousal = nn.MSELoss()            # per regressione arousal

    # Ottimizzatore
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    model.LAYER_RoBERTa.eval()
    for param in model.LAYER_RoBERTa.parameters():
        param.requires_grad = False

    try:
        print("Starting training... ")
        total_time = 0.0
        for epoch in range(epochs):
            total_loss = 0.0
            time_start = time.time()
            for batch_texts, batch_valence, batch_arousal in dataloader:
                # Sposta su device
                batch_valence = batch_valence.to(device)
                batch_arousal = batch_arousal.to(device)

                # Forward
                valence_pred, arousal_pred = model.forward_heads_only(batch_texts,device)

                # CrossEntropyLoss richiede target come classe int, non one-hot
                target_valence = batch_valence.argmax(dim=1)

                loss1 = loss_valence(valence_pred, target_valence)
                loss2 = loss_arousal(arousal_pred.squeeze(), batch_arousal)

                loss = loss1 + loss2  # somma le due loss per multi-task

                # Backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            time_end = time.time()
            total_time += (time_end-time_start)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}, Time: {time_end-time_start:.4f} s")
        print(f"Training Finish! Time: {total_time}")
    
    except Exception as e:
        print(f"Exception {e}")

    torch.save(model.state_dict(), f"{path_save}/{name_model_save}")
    return model





