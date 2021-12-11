# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_DWA_Simulator(object):
    def setupUi(self, DWA_Simulator):
        DWA_Simulator.setObjectName("DWA_Simulator")
        DWA_Simulator.resize(1012, 670)
        self.graphicsView = QtWidgets.QGraphicsView(DWA_Simulator)
        self.graphicsView.setGeometry(QtCore.QRect(20, 50, 600, 600))
        self.graphicsView.setObjectName("graphicsView")
        self.Start_simulation_pushButton = QtWidgets.QPushButton(DWA_Simulator)
        self.Start_simulation_pushButton.setGeometry(QtCore.QRect(630, 620, 170, 30))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.Start_simulation_pushButton.setFont(font)
        self.Start_simulation_pushButton.setObjectName("Start_simulation_pushButton")
        self.Pause_pushButton = QtWidgets.QPushButton(DWA_Simulator)
        self.Pause_pushButton.setGeometry(QtCore.QRect(810, 620, 75, 30))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.Pause_pushButton.setFont(font)
        self.Pause_pushButton.setObjectName("Pause_pushButton")
        self.Reset_pushButton = QtWidgets.QPushButton(DWA_Simulator)
        self.Reset_pushButton.setGeometry(QtCore.QRect(895, 620, 81, 30))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.Reset_pushButton.setFont(font)
        self.Reset_pushButton.setObjectName("Reset_pushButton")
        self.Possible_passes_spinBox = QtWidgets.QSpinBox(DWA_Simulator)
        self.Possible_passes_spinBox.setGeometry(QtCore.QRect(940, 580, 60, 25))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.Possible_passes_spinBox.setFont(font)
        self.Possible_passes_spinBox.setMinimum(2)
        self.Possible_passes_spinBox.setObjectName("Possible_passes_spinBox")
        self.label = QtWidgets.QLabel(DWA_Simulator)
        self.label.setGeometry(QtCore.QRect(630, 580, 261, 21))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.Max_Acc_SpinBox = QtWidgets.QDoubleSpinBox(DWA_Simulator)
        self.Max_Acc_SpinBox.setGeometry(QtCore.QRect(940, 360, 60, 25))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.Max_Acc_SpinBox.setFont(font)
        self.Max_Acc_SpinBox.setSingleStep(0.05)
        self.Max_Acc_SpinBox.setObjectName("Max_Acc_SpinBox")
        self.label_2 = QtWidgets.QLabel(DWA_Simulator)
        self.label_2.setGeometry(QtCore.QRect(630, 360, 221, 21))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(DWA_Simulator)
        self.label_3.setGeometry(QtCore.QRect(630, 390, 301, 21))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.Max_Ang_Acc_SpinBox = QtWidgets.QDoubleSpinBox(DWA_Simulator)
        self.Max_Ang_Acc_SpinBox.setGeometry(QtCore.QRect(940, 390, 60, 25))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.Max_Ang_Acc_SpinBox.setFont(font)
        self.Max_Ang_Acc_SpinBox.setSingleStep(0.05)
        self.Max_Ang_Acc_SpinBox.setObjectName("Max_Ang_Acc_SpinBox")
        self.Max_Vel_SpinBox = QtWidgets.QDoubleSpinBox(DWA_Simulator)
        self.Max_Vel_SpinBox.setGeometry(QtCore.QRect(940, 420, 60, 25))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.Max_Vel_SpinBox.setFont(font)
        self.Max_Vel_SpinBox.setSingleStep(0.05)
        self.Max_Vel_SpinBox.setObjectName("Max_Vel_SpinBox")
        self.Min_Vel_SpinBox = QtWidgets.QDoubleSpinBox(DWA_Simulator)
        self.Min_Vel_SpinBox.setGeometry(QtCore.QRect(940, 450, 60, 25))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.Min_Vel_SpinBox.setFont(font)
        self.Min_Vel_SpinBox.setSingleStep(0.05)
        self.Min_Vel_SpinBox.setObjectName("Min_Vel_SpinBox")
        self.label_4 = QtWidgets.QLabel(DWA_Simulator)
        self.label_4.setGeometry(QtCore.QRect(630, 420, 221, 21))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(DWA_Simulator)
        self.label_5.setGeometry(QtCore.QRect(630, 450, 191, 21))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(DWA_Simulator)
        self.label_6.setGeometry(QtCore.QRect(630, 480, 261, 21))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(DWA_Simulator)
        self.label_7.setGeometry(QtCore.QRect(630, 510, 251, 21))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.Max_Ang_Vel_SpinBox = QtWidgets.QDoubleSpinBox(DWA_Simulator)
        self.Max_Ang_Vel_SpinBox.setGeometry(QtCore.QRect(940, 480, 60, 25))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.Max_Ang_Vel_SpinBox.setFont(font)
        self.Max_Ang_Vel_SpinBox.setSingleStep(0.05)
        self.Max_Ang_Vel_SpinBox.setObjectName("Max_Ang_Vel_SpinBox")
        self.Min_Ang_Vel_SpinBox = QtWidgets.QDoubleSpinBox(DWA_Simulator)
        self.Min_Ang_Vel_SpinBox.setGeometry(QtCore.QRect(940, 510, 60, 25))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.Min_Ang_Vel_SpinBox.setFont(font)
        self.Min_Ang_Vel_SpinBox.setMinimum(-99.9)
        self.Min_Ang_Vel_SpinBox.setSingleStep(0.05)
        self.Min_Ang_Vel_SpinBox.setObjectName("Min_Ang_Vel_SpinBox")
        self.label_8 = QtWidgets.QLabel(DWA_Simulator)
        self.label_8.setGeometry(QtCore.QRect(630, 320, 331, 31))
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setBold(True)
        font.setWeight(75)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.vel_delta_SpinBox = QtWidgets.QDoubleSpinBox(DWA_Simulator)
        self.vel_delta_SpinBox.setGeometry(QtCore.QRect(940, 107, 60, 25))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.vel_delta_SpinBox.setFont(font)
        self.vel_delta_SpinBox.setSingleStep(0.01)
        self.vel_delta_SpinBox.setObjectName("vel_delta_SpinBox")
        self.ang_vel_delta_SpinBox = QtWidgets.QDoubleSpinBox(DWA_Simulator)
        self.ang_vel_delta_SpinBox.setGeometry(QtCore.QRect(940, 137, 60, 25))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.ang_vel_delta_SpinBox.setFont(font)
        self.ang_vel_delta_SpinBox.setSingleStep(0.01)
        self.ang_vel_delta_SpinBox.setObjectName("ang_vel_delta_SpinBox")
        self.sampling_interval_SpinBox = QtWidgets.QDoubleSpinBox(DWA_Simulator)
        self.sampling_interval_SpinBox.setGeometry(QtCore.QRect(940, 167, 60, 25))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.sampling_interval_SpinBox.setFont(font)
        self.sampling_interval_SpinBox.setSingleStep(0.1)
        self.sampling_interval_SpinBox.setObjectName("sampling_interval_SpinBox")
        self.label_9 = QtWidgets.QLabel(DWA_Simulator)
        self.label_9.setGeometry(QtCore.QRect(630, 47, 121, 21))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label_9.setFont(font)
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(DWA_Simulator)
        self.label_10.setGeometry(QtCore.QRect(630, 77, 131, 21))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")
        self.label_11 = QtWidgets.QLabel(DWA_Simulator)
        self.label_11.setGeometry(QtCore.QRect(630, 107, 161, 21))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label_11.setFont(font)
        self.label_11.setObjectName("label_11")
        self.label_12 = QtWidgets.QLabel(DWA_Simulator)
        self.label_12.setGeometry(QtCore.QRect(630, 137, 161, 21))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label_12.setFont(font)
        self.label_12.setObjectName("label_12")
        self.label_13 = QtWidgets.QLabel(DWA_Simulator)
        self.label_13.setGeometry(QtCore.QRect(630, 167, 191, 21))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label_13.setFont(font)
        self.label_13.setObjectName("label_13")
        self.pre_time_spinBox = QtWidgets.QSpinBox(DWA_Simulator)
        self.pre_time_spinBox.setGeometry(QtCore.QRect(940, 47, 60, 25))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.pre_time_spinBox.setFont(font)
        self.pre_time_spinBox.setMinimum(2)
        self.pre_time_spinBox.setObjectName("pre_time_spinBox")
        self.pre_step_spinBox = QtWidgets.QSpinBox(DWA_Simulator)
        self.pre_step_spinBox.setGeometry(QtCore.QRect(940, 77, 60, 25))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.pre_step_spinBox.setFont(font)
        self.pre_step_spinBox.setMinimum(2)
        self.pre_step_spinBox.setObjectName("pre_step_spinBox")
        self.label_14 = QtWidgets.QLabel(DWA_Simulator)
        self.label_14.setGeometry(QtCore.QRect(630, 197, 191, 21))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label_14.setFont(font)
        self.label_14.setObjectName("label_14")
        self.label_15 = QtWidgets.QLabel(DWA_Simulator)
        self.label_15.setGeometry(QtCore.QRect(630, 227, 191, 21))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label_15.setFont(font)
        self.label_15.setObjectName("label_15")
        self.label_16 = QtWidgets.QLabel(DWA_Simulator)
        self.label_16.setGeometry(QtCore.QRect(630, 257, 191, 21))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label_16.setFont(font)
        self.label_16.setObjectName("label_16")
        self.angle_weight_SpinBox = QtWidgets.QDoubleSpinBox(DWA_Simulator)
        self.angle_weight_SpinBox.setGeometry(QtCore.QRect(940, 197, 60, 25))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.angle_weight_SpinBox.setFont(font)
        self.angle_weight_SpinBox.setSingleStep(0.01)
        self.angle_weight_SpinBox.setObjectName("angle_weight_SpinBox")
        self.velocity_weight_SpinBox = QtWidgets.QDoubleSpinBox(DWA_Simulator)
        self.velocity_weight_SpinBox.setGeometry(QtCore.QRect(940, 227, 60, 25))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.velocity_weight_SpinBox.setFont(font)
        self.velocity_weight_SpinBox.setSingleStep(0.01)
        self.velocity_weight_SpinBox.setObjectName("velocity_weight_SpinBox")
        self.obstacle_weight_SpinBox = QtWidgets.QDoubleSpinBox(DWA_Simulator)
        self.obstacle_weight_SpinBox.setGeometry(QtCore.QRect(940, 257, 60, 25))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.obstacle_weight_SpinBox.setFont(font)
        self.obstacle_weight_SpinBox.setSingleStep(0.01)
        self.obstacle_weight_SpinBox.setObjectName("obstacle_weight_SpinBox")
        self.label_17 = QtWidgets.QLabel(DWA_Simulator)
        self.label_17.setGeometry(QtCore.QRect(631, 11, 211, 31))
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setBold(True)
        font.setWeight(75)
        self.label_17.setFont(font)
        self.label_17.setObjectName("label_17")
        self.label_18 = QtWidgets.QLabel(DWA_Simulator)
        self.label_18.setGeometry(QtCore.QRect(629, 544, 231, 31))
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setBold(True)
        font.setWeight(75)
        self.label_18.setFont(font)
        self.label_18.setObjectName("label_18")
        self.area_dis_to_obs_SpinBox = QtWidgets.QDoubleSpinBox(DWA_Simulator)
        self.area_dis_to_obs_SpinBox.setGeometry(QtCore.QRect(940, 287, 60, 25))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.area_dis_to_obs_SpinBox.setFont(font)
        self.area_dis_to_obs_SpinBox.setSingleStep(0.01)
        self.area_dis_to_obs_SpinBox.setObjectName("area_dis_to_obs_SpinBox")
        self.label_19 = QtWidgets.QLabel(DWA_Simulator)
        self.label_19.setGeometry(QtCore.QRect(630, 287, 291, 21))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label_19.setFont(font)
        self.label_19.setObjectName("label_19")

        self.retranslateUi(DWA_Simulator)
        self.Start_simulation_pushButton.clicked.connect(DWA_Simulator.Start_simulation)
        self.Pause_pushButton.clicked.connect(DWA_Simulator.pause)
        self.Reset_pushButton.clicked.connect(DWA_Simulator.reset)
        QtCore.QMetaObject.connectSlotsByName(DWA_Simulator)

    def retranslateUi(self, DWA_Simulator):
        _translate = QtCore.QCoreApplication.translate
        DWA_Simulator.setWindowTitle(_translate("DWA_Simulator", "DWA SIMULATOR"))
        self.Start_simulation_pushButton.setText(_translate("DWA_Simulator", "Start simulation"))
        self.Pause_pushButton.setText(_translate("DWA_Simulator", "Pause"))
        self.Reset_pushButton.setText(_translate("DWA_Simulator", "Reset"))
        self.label.setText(_translate("DWA_Simulator", "Number of possible passes"))
        self.label_2.setText(_translate("DWA_Simulator", "Maximum acceleration"))
        self.label_3.setText(_translate("DWA_Simulator", "Maximum angular acceleration"))
        self.label_4.setText(_translate("DWA_Simulator", "Maximum velocity"))
        self.label_5.setText(_translate("DWA_Simulator", "Minimum velocity"))
        self.label_6.setText(_translate("DWA_Simulator", "Maximum angular velocity"))
        self.label_7.setText(_translate("DWA_Simulator", "Minimum angular velocity"))
        self.label_8.setText(_translate("DWA_Simulator", "Two wheeld robot parameters"))
        self.label_9.setText(_translate("DWA_Simulator", "pre time"))
        self.label_10.setText(_translate("DWA_Simulator", "pre step"))
        self.label_11.setText(_translate("DWA_Simulator", "delta vel"))
        self.label_12.setText(_translate("DWA_Simulator", "delta ang vel"))
        self.label_13.setText(_translate("DWA_Simulator", "sampling interval"))
        self.label_14.setText(_translate("DWA_Simulator", "angle weight"))
        self.label_15.setText(_translate("DWA_Simulator", "velocity weight"))
        self.label_16.setText(_translate("DWA_Simulator", "obstacle weight"))
        self.label_17.setText(_translate("DWA_Simulator", "DWA parameters"))
        self.label_18.setText(_translate("DWA_Simulator", "Drawing parameters"))
        self.label_19.setText(_translate("DWA_Simulator", "distance to obsticle threshold"))