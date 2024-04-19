from PIL import Image 
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm.notebook import tqdm
from sklearn.metrics import f1_score
import numpy as np
import torch
import gc

class CustomDataset(Dataset):

    def __init__(self, path):
        self.path = path
          
    def __getitem__(self, idx):
        img = np.array(Image.open(self.path).resize((224, 224)), dtype='uint8')
        
        
        T = v2.Compose(
            [#transforms.ToTensor(),
             v2.ToImage(),
             v2.ToDtype(torch.uint8, scale=True),
             v2.Resize((224, 224), antialias=True),
             v2.ToDtype(torch.float32, scale=True),
             v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
             ] 
        )

        img = T(img)
        
        return {'image': img}
               
    def __len__(self):
        return 1
    

def evaluation_cv(img_path):  
    test = CustomDataset(img_path)

    test_dataloader = DataLoader(test, batch_size=1)

    resnet = torchvision.models.resnet18()
    resnet.fc = torch.nn.Linear(in_features= 512, out_features=2, bias=True)
    checkpoint = torch.load('/home/rijkaa/leraa/solution/best_model_artifact_250000.pt')
    for key in list(checkpoint.keys()):
        if key.startswith('module.'):
            checkpoint[key[7:]] = checkpoint.pop(key)
    
    resnet.load_state_dict(checkpoint)
    
    #resnet.load_state_dict('/home/rijkaa/leraa/solution/best_model_artifact_250000.pt') # написать путь к чекпоинту
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    resnet.to(device)
    resnet.eval()
    with torch.no_grad():
        for data in tqdm(test_dataloader):
            img = data["image"].cuda()
            prediction = resnet(img).argmax(dim = 1).cpu().detach().numpy()
            
            if prediction[0] > 0.5:
                print('Generated')
            else:
                print('Natural')
            


    torch.cuda.empty_cache()
    gc.collect()