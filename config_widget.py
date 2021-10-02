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

# ui配置文件
cUi, cBase = uic.loadUiType("config_widget.ui")

# 主界面
class CConfigWidget(QWidget, cUi):
    def __init__(self, main_widget=None):
        # 设置UI
        QMainWindow.__init__(self)
        cUi.__init__(self)
        self.setupUi(self)
        self.main_widget = main_widget
        self.m_prj = None
        self.config_map = None
        self.edit_map = None
        self.simple_keys = ['yolox_root', 'num_classes', 'depth', 'width', 'input_size', 'data_dir', 'max_epoch', 
                            'basic_lr_per_img', 'train_ann', 'val_ann', 'enable_mixup', 'test_size', 
                            'test_conf', 'nmsthre']
        self.show_mode = 'simple'
        
    def check_import_config(self):
        yolox_root = self.get_config_value("yolox_root").replace('"', '')
        data_dir = self.get_config_value("data_dir").replace('"', '')
        print('--->', yolox_root)
        if not os.path.exists(yolox_root):
            return "yolox_root is not exist"
        if not os.path.exists(data_dir):
            return "data_dir is not exist"
        return "ok"

    def check_train_status(self):
        #if self.m_prj is not None:
        #    if self.m_prj
        pass
    
    def set_prj(self, prj_obj):
        self.m_prj = prj_obj
        self.show_prj_config()
    
    def show_prj_config(self):
        self.config_map = self.m_prj.get_exp()
        self.edit_map = {}
        for i in range(self.layout.count()):
            widget = self.layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()
        #while self.layout.count() > 0:
        #    self.layout.removeItem(self.layout.itemAt(0))

        for key in self.config_map.keys():
            group_box = QGroupBox()
            group_box.setTitle(key)
            group_layout = QVBoxLayout()
            for sub_key in self.config_map[key].keys():
                if self.show_mode == 'simple':
                    if sub_key not in self.simple_keys:
                        continue
                sub_value = self.config_map[key][sub_key]
                edit_layout = QHBoxLayout()
                edit_key = QLineEdit()
                edit_value = QLineEdit()
                if key not in self.edit_map.keys():
                    self.edit_map[key] = {}
                self.edit_map[key][sub_key] = edit_value
                edit_key.setText(sub_key)
                edit_key.setReadOnly(False)
                edit_key.setFocusPolicy(Qt.NoFocus)
                edit_value.setText(str(sub_value))
                edit_layout.addWidget(edit_key)
                edit_layout.addWidget(edit_value)
                group_layout.addLayout(edit_layout)
            group_box.setLayout(group_layout)  
            self.layout.addWidget(group_box)
        spacerItem = PyQt5.QtWidgets.QSpacerItem(20, 20, PyQt5.QtWidgets.QSizePolicy.Minimum,
                PyQt5.QtWidgets.QSizePolicy.Expanding)
        self.layout.addItem(spacerItem)
        
    def save_prj_config(self):
        for key in self.edit_map.keys():
            for sub_key in self.edit_map[key].keys():
                edit_value = self.edit_map[key][sub_key]
                self.config_map[key][sub_key] = edit_value.text()
        #print(self.config_map)

        self.m_prj.save_exp(self.config_map)

        check_info = self.check_import_config()
        if check_info != 'ok':
            reply = QMessageBox.warning(self,
                  u'错误', 
                  check_info,
                  QMessageBox.Yes)
            return
    
    def get_config_value(self, key):
        for root_key in self.config_map.keys():
            for sub_key in self.config_map[root_key].keys():
                #print('@',sub_key, '<->', key)
                if sub_key == key:
                    return self.config_map[root_key][sub_key]
        return ""
        

    @pyqtSlot()
    def on_btnSave_clicked(self):
        print('button save clicked')
        self.save_prj_config()
        
    @pyqtSlot()
    def on_btnReset_clicked(self):
        print('button reset clicked')
        self.m_prj.reset_exp()
        self.show_prj_config()

    def on_radioSimple_toggled(self):
        if self.radioSimple.isChecked():
            print('simple push')
            self.show_mode = 'simple'
            self.show_prj_config()
        
    def on_radioDetail_toggled(self):
        if self.radioDetail.isChecked():
            print('deatail push')
            self.show_mode = 'detail'
            self.show_prj_config()

if __name__ == "__main__":
    cApp = QApplication(sys.argv)
    
    m_prj = ZXProj("xxx")
    m_prj.create_proj()
    
    cConfigWidget = CConfigWidget()
    cConfigWidget.set_prj(m_prj)
    cConfigWidget.show()
    sys.exit(cApp.exec_())