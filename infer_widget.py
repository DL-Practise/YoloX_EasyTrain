# -*- coding: utf-8 -*-
import sys
import os
import PyQt5
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui, uic
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from api import ZXProj
import copy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import matplotlib
import matplotlib.cbook as cbook
import numpy as np
import psutil



# ui配置文件
cUi, cBase = uic.loadUiType("infer_widget.ui")

# 主界面
class CInferWidget(QWidget, cUi):
    def __init__(self, main_widget=None):
        # 设置UI
        QMainWindow.__init__(self)
        cUi.__init__(self)
        self.setupUi(self)
        self.main_widget = main_widget
        self.m_prj = None
        
    def set_prj(self, prj_obj):
        self.m_prj = prj_obj

    def result_cb(self, result_path):
        image = QImage(result_path)
        #self.labelShow.setAlignment(Qt.AlignCenter)
        self.labelShow.setScaledContents(True)
        self.labelShow.setPixmap(QPixmap.fromImage(image))

    @pyqtSlot()
    def on_btnModel_clicked(self):
        print('button btnModel clicked: ', os.path.join(os.getcwd(), self.m_prj.model_dir))
        pretrain=QFileDialog.getOpenFileName(self, '选择预训练模型', os.path.join(os.getcwd(), self.m_prj.model_dir), "pth model(*.pth);;pt model(*.pt)")
        if pretrain[0] == '':
            print('no pretrain model select')
        else:
            print('select pretrain model')
            self.editModel.setText(pretrain[0])

    @pyqtSlot()
    def on_btnImg_clicked(self):
        print('button btnImg clicked')
        img_path=QFileDialog.getOpenFileName(self, '选择预图片', os.getcwd(), "jpg(*.jpg);;JPG(*.JPG);;png(*.png);;PNG(*.PNG)")
        if img_path[0] == '':
            print('no img select')
        else:
            print('img select')
            self.editImg.setText(img_path[0])

    @pyqtSlot()
    def on_btnInfer_clicked(self):
        print('button btnInfer clicked')
        self.m_prj.infer_img(self.editModel.text(), 
                             self.editImg.text(), 
                             self.editConf.text(), 
                             self.editNms.text(), 
                             self.editSize.text(),
                             self.result_cb)
        
 
if __name__ == "__main__":
    cApp = QApplication(sys.argv)
    
    m_prj = ZXProj("xxx")
    m_prj.create_proj()
    
    cInferWidget = CInferWidget()
    cInferWidget.set_prj(m_prj)
    cInferWidget.show()
    sys.exit(cApp.exec_())