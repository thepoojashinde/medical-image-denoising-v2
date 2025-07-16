import numpy as np
import cv2
import matplotlib.pyplot as plt
import pydicom as dicom

class dicom_processor:

    def __init__(self,img_path):
        self.img_path = img_path
        self.ds = None
        self.img = None
        self.img_uint8 = None
        self.load_dicom()


    def load_dicom(self):
      try:
         self.ds = dicom.dcmread(self.img_path)
         self.img = self.ds.pixel_array.astype(np.float32)
         #normalize
         self.img = cv2.normalize(self.img,None,0,255,cv2.NORM_MINMAX)
         self.img_uint8 = self.img.astype(np.uint8)
      except:
          self.img = None
          self.img_uint8 = None

    def get_metadata(self):
        return{
            'PatientName': self.ds.get('PatientName'),
            'Modality': self.ds.get('Modality'),
            'StudyDate': self.ds.get('StudyDate'),
            'Size': self.img.shape
        }

    def get_image(self):
        return self.img

    def display_image_plt(self):
        plt.imshow(self.img,cmap = 'gray')
        plt.title('DICOM IMAGE')
        plt.axis('on')
        plt.show()

    def display_image_cv2(self):
        cv2.imshow('DICOM IMAGE',self.img_uint8)
        cv2.waitKey(0)

    def get_clean_image(self):
        return self.img_uint8

    def get_noisy_image(self,noise_type = 'gaussian',std_dev=20):
        if self.img_uint8 is None:
            return None
        if noise_type == 'gaussian':
            noise = np.random.normal(0,std_dev,self.img_uint8.shape)
            noisy = self.img_uint8.astype(np.float32)+noise
        elif noise_type == 'poisson':
            vals = 2**np.ceil(np.log2(len(np.unique(self.img_uint8))))
            nosiy = np.random.poisson(self.img_uint8*vals)/float(vals)
        else:
            return None

        noisy = np.clip(noisy,0,255).astype(np.uint8)
        return noisy

    def prepare_for_model(self,img):
        resized = cv2.resize(img,(128,128)).astype(np.float32)/255.0
        return np.expand_dims(resized,-1)






dcm = dicom_processor('lidc_subject_0001_slice_40.dcm')
info = dcm.get_metadata()
print(info)
dcm.display_image_plt()

