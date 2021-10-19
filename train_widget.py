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
cUi, cBase = uic.loadUiType("train_widget.ui")

# 主界面
class CTrainWidget(QWidget, cUi):
    process_sig = pyqtSignal(list)

    def __init__(self, main_widget=None):
        # 设置UI
        QMainWindow.__init__(self)
        cUi.__init__(self)
        self.setupUi(self)
        self.main_widget = main_widget
        self.m_prj = None
        self.progressBar.setValue(0)

        self.showFigure = Figure_Canvas()
 
        layout = QGridLayout()
        layout.addWidget(self.showFigure)
        self.frame.setLayout(layout)

        self.process_sig.connect(self.update_process_bar)

    def set_prj(self, prj_obj):
        self.m_prj = prj_obj
        self.m_prj.train_cb = self.info_cb

    def info_cb(self, info):
        if 'iter_info' in info.keys() or 'map' in info.keys():
            self.showFigure.add_data(info)
        if 'progress_info' in info.keys():
            #self.update_process_bar(info['progress_info'])
            self.process_sig.emit(info['progress_info'])
        if 'train_status' in info.keys():
            if self.m_prj.is_train():
                self.m_prj.stop_train()
                self.btnTrain.setText("开始训练")
        if 'train_status' in info.keys():
            if info['train_status'] == 'finished':
                self.m_prj.stop_train()
                self.btnTrain.setText("开始训练")
        #if 'log' in info.keys():
        #    self.textLog.append(info['log'])
            
    def update_process_bar(self, infos):
        progress_value, iter_time, progress_str, remain_time = infos

        gpu_count = float(self.editGpu.text())
        batch_size = float(self.editBatch.text())
        speed = '%.2f imgs/sec'%(gpu_count * batch_size / iter_time)
        self.progressBar.setValue(int(progress_value * 100))
        self.progressBar.setFormat(u'speed:%s %s 剩余时间:%s'%(speed, progress_str, remain_time))

    @pyqtSlot()
    def on_btnPretrain_clicked(self):
        print('button btnPretrain clicked')
        pretrain=QFileDialog.getOpenFileName(self, '选择预训练模型', os.getcwd() + '/weights', "pth model(*.pth);;pt model(*.pt)")
        if pretrain[0] == '':
            print('no pretrain model select')
        else:
            print('select pretrain model')
        self.editPretrain.setText(pretrain[0])

    @pyqtSlot()
    def on_btnTrain_clicked(self):
        print('button train clicked')
        if self.m_prj.is_train():
            self.m_prj.stop_train()
            self.btnTrain.setText("开始训练")
        else:
            gpu_count = self.editGpu.text()
            batch_size = self.editBatch.text()
            pretrain = self.editPretrain.text()
            if self.radioFp16.isChecked():
                mix_train = True
            else:
                mix_train = False
            self.m_prj.start_train(gpu_count, batch_size, mix_train, pretrain, self.info_cb)
            self.btnTrain.setText("停止训练")
            self.showFigure.clear_draw()

    def on_editRate_textChanged(self):
        print(self.editRate.text())
        
class Figure_Canvas(FigureCanvas):
    def __init__(self,parent=None,width=3.9,height=2.7,dpi=100):
        self.fig=Figure(figsize=(3.9, 2.7), dpi=100)
        super(Figure_Canvas,self).__init__(self.fig)
        #self.setParent(parent)
        self.ax_total_loss=self.fig.add_subplot(231)
        self.ax_total_loss.set_title("total loss")
        self.ax_iou_loss=self.fig.add_subplot(232)
        self.ax_iou_loss.set_title("iou loss")
        #self.ax_l1_loss=self.fig.add_subplot(233)
        #self.ax_l1_loss.set_title("l1 loss")
        self.ax_conf_loss=self.fig.add_subplot(233)
        self.ax_conf_loss.set_title("conf loss")
        self.ax_cls_loss=self.fig.add_subplot(234)
        self.ax_cls_loss.set_title("cls loss")
        self.ax_lr=self.fig.add_subplot(235)
        self.ax_lr.set_title("lr")
        self.ax_map=self.fig.add_subplot(236)
        self.ax_map.set_title("map")
        self.fig.subplots_adjust(hspace=0.3)

        self.iters = []
        self.total_losses = []
        self.iou_losses = []
        #self.l1_losses = []
        self.conf_losses = []
        self.cls_losses = []
        self.lrs = []
        self.epochs = []
        self.map_50 = []
        self.map_50_95 = []
        self.map_75 = []

    def add_data(self, add_data): 
        if 'iter_info' in add_data.keys():
            self.iters.append(add_data['iter_info'])
        if 'loss_info' in add_data.keys():
            self.total_losses.append(add_data['loss_info'][0])
            self.iou_losses.append(add_data['loss_info'][1])
            self.conf_losses.append(add_data['loss_info'][3])
            self.cls_losses.append(add_data['loss_info'][4])
        if 'lr_info' in add_data.keys():
            self.lrs.append(add_data['lr_info'])
        if 'epoch' in add_data.keys():
            self.epochs.append(add_data['epoch'])
        if 'map' in add_data.keys():
            self.map_50.append(add_data['map'][0])
            self.map_50_95.append(add_data['map'][2])
            self.map_75.append(add_data['map'][1])
        self.update_draw()

    def clear_draw(self):
        self.ax_total_loss.clear()
        self.ax_iou_loss.clear()
        self.ax_conf_loss.clear()
        self.ax_cls_loss.clear()
        self.ax_lr.clear()
        self.ax_map.clear()
        self.draw()

    def update_draw(self):
        self.ax_total_loss.clear()
        self.ax_total_loss.set_title("total loss")
        self.ax_total_loss.plot(self.iters, self.total_losses)
        self.ax_iou_loss.clear()
        self.ax_iou_loss.set_title("iou loss")
        self.ax_iou_loss.plot(self.iters, self.iou_losses)
        #self.ax_l1_loss.clear()
        #self.ax_l1_loss.set_title("l1 loss")
        #self.ax_l1_loss.plot(self.iters, self.l1_losses)
        self.ax_conf_loss.clear()
        self.ax_conf_loss.set_title("conf loss")
        self.ax_conf_loss.plot(self.iters, self.conf_losses)
        self.ax_cls_loss.clear()
        self.ax_cls_loss.set_title("cls loss")
        self.ax_cls_loss.plot(self.iters, self.cls_losses)
        self.ax_lr.clear()
        self.ax_lr.set_title("lr")
        self.ax_lr.plot(self.iters, self.lrs)
        self.ax_map.clear()
        self.ax_map.set_title("map")
        line_map50 = self.ax_map.plot(self.epochs, self.map_50, label='map50')
        line_map50_95 = self.ax_map.plot(self.epochs, self.map_50_95, label='map50-95')
        line_map75 = self.ax_map.plot(self.epochs, self.map_75, label='map75')
        handles, labels = self.ax_map.get_legend_handles_labels()
        self.ax_map.legend(handles, labels)
        #self.ax_map.legend(handles=[line_map50,line_map50_95,line_map75],labels=['map50', 'map50-95', 'map75'],loc='best')
        self.draw()

if __name__ == "__main__":
    cApp = QApplication(sys.argv)
    
    m_prj = ZXProj("xxx")
    m_prj.create_proj()
    
    cTrainWidget = CTrainWidget()
    cTrainWidget.set_prj(m_prj)
    cTrainWidget.show()
    sys.exit(cApp.exec_())