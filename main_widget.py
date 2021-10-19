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
import numpy as np
from config_widget import CConfigWidget
from train_widget import CTrainWidget
from infer_widget import CInferWidget

# ui配置文件
cUi, cBase = uic.loadUiType("main_widget.ui")

# 主界面
class CMainWidget(QWidget, cUi):
    def __init__(self, main_widget=None):
        # 设置UI
        QMainWindow.__init__(self)
        cUi.__init__(self)
        self.setupUi(self)
        self.setWindowTitle(u'YoloX 可视化训练插件')
        self.main_widget = main_widget

        self.config_widget = CConfigWidget()
        self.train_widget = CTrainWidget()
        self.infer_widget = CInferWidget()
        self.tabWidget.addTab(self.config_widget, u"配置")
        self.tabWidget.addTab(self.train_widget, u"训练")
        self.tabWidget.addTab(self.infer_widget, u"推理")
        self.m_proj = None

        self.on_treeProj_init()

    def on_treeProj_init(self):
        self.treeProj.header().setVisible(False)
        ignore_dirs = ['backup', '__pycache__', ]
        root_dir = './projects/'
        for prj in os.listdir(root_dir):
            if os.path.isdir('./projects/'+prj) and str(prj) not in ignore_dirs:
                item = QTreeWidgetItem(self.treeProj)
                item.setText(0, str(prj))
        print(self.treeProj.topLevelItemCount())
        
    def on_treeProj_itemClicked(self, item, seq):
        if self.m_proj is not None and self.m_proj.is_train():
            reply = QMessageBox.warning(self,
                  u'警告', 
                  u'当前工程正在训练，无法切换工程', 
                  QMessageBox.Yes)
            old_prj_name = self.m_proj.prj_name
            iterator = QTreeWidgetItemIterator(self.treeProj)
            while iterator.value():
                item = iterator.value()
                if item.text(0) == old_prj_name:
                    self.treeProj.setCurrentItem(item)
                    break   
                iterator.__iadd__(1) 
        else:
            prj_name = item.text(0)
            self.change_proj(prj_name)

    def change_proj(self, proj_name, model_name=None):
        self.m_proj = ZXProj(proj_name, model_name)
        self.config_widget.set_prj(self.m_proj)
        self.train_widget.set_prj(self.m_proj)
        self.infer_widget.set_prj(self.m_proj)
        title_name = u'YoloX 可视化训练插件 当前工程：' + self.m_proj.prj_name
        self.setWindowTitle(title_name)
        self.tabWidget.setCurrentIndex(1)
        return True

    @pyqtSlot()
    def on_btnAddPrj_clicked(self):
        print('button btnAddPrj clicked')
        if self.m_proj is not None and self.m_proj.is_train():
            reply = QMessageBox.warning(self,
                  u'警告', 
                  u'当前工程正在训练，无法新建工程', 
                  QMessageBox.Yes)
            return
        else:
            text, okPressed = QInputDialog.getText(self, "请输入工程名称","工程名（英文且不包含空格等特殊字符）:", QLineEdit.Normal, "")
            if okPressed and text != '':
                model_names=('yolox_s','yolox_m','yolox_l','yolox_tiny','yolox_nano')
                model_name,ok=QInputDialog.getItem(self,"选择基线模型","基线模型",model_names,0,False)
                if ok and model_name:
                    proj_show_name = text + ':' +  model_name
                    iterator = QTreeWidgetItemIterator(self.treeProj)
                    while iterator.value():
                        item = iterator.value()
                        if item.text(0) == proj_show_name:
                            reply = QMessageBox.warning(self,
                                u'警告', 
                                u'该工程已经存在', 
                                QMessageBox.Yes)
                            return   
                        iterator.__iadd__(1) 

                    item = QTreeWidgetItem(self.treeProj)
                    item.setText(0, text)
                    self.treeProj.setCurrentItem(item)
                    self.change_proj(text, model_name)

    @pyqtSlot()
    def on_btnDelPrj_clicked(self):
        print('del')
        print('button btnAddPrj clicked')
        if self.m_proj is not None and self.m_proj.is_train():
            reply = QMessageBox.warning(self,
                  u'警告', 
                  u'当前工程正在训练，无法删除工程', 
                  QMessageBox.Yes)
            return
        else:
            reply = QMessageBox.warning(self,
                  u'警告', 
                  u'删除该工程%s后将不可恢复，确定吗'%self.m_proj.prj_name, 
                  QMessageBox.Yes | QMessageBox.No)
            if reply==QMessageBox.Yes:
                for item in self.treeProj.selectedItems():
                    index = self.treeProj.indexOfTopLevelItem(item)
                    self.treeProj.takeTopLevelItem(index)
                    break
                self.m_proj.delete_proj()
            else:
                return

    def closeEvent(self,event):
        if self.m_proj is not None and self.m_proj.is_train():
            reply = QMessageBox.warning(self,
                  u'警告', 
                  u'训练任务尚无退出，请先停止训练', 
                  QMessageBox.Yes)
            event.ignore()
        else:
            reply = QMessageBox.warning(self,
                  u'警告', 
                  u'确认退出?', 
                  QMessageBox.Yes | QMessageBox.No)
            if reply==QMessageBox.Yes:
                self.config_widget.save_prj_config()
                event.accept()
            else:
                event.ignore()

if __name__ == "__main__":
    cApp = QApplication(sys.argv)
    cMainWidget = CMainWidget()
    cMainWidget.show()
    sys.exit(cApp.exec_())