# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'face.ui'
##
## Created by: Qt User Interface Compiler version 6.9.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QGroupBox, QLabel, QLineEdit,
    QMainWindow, QMenuBar, QPlainTextEdit, QPushButton,
    QSizePolicy, QStatusBar, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(952, 759)
        font = QFont()
        font.setPointSize(18)
        MainWindow.setFont(font)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.label = QLabel(self.centralwidget)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(240, 20, 491, 71))
        font1 = QFont()
        font1.setFamilies([u"Segoe Script"])
        font1.setPointSize(26)
        self.label.setFont(font1)
        self.label.setStyleSheet(u"color: rgb(0, 8, 255);\n"
"background-color: rgb(255,255,255);")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.groupBox = QGroupBox(self.centralwidget)
        self.groupBox.setObjectName(u"groupBox")
        self.groupBox.setGeometry(QRect(80, 120, 181, 441))
        self.groupBox.setStyleSheet(u"border-color: rgb(199, 255, 220);\n"
"border-top-color: rgb(138, 255, 146);\n"
"background-color: rgb(185, 246, 255);")
        self.pushButton = QPushButton(self.groupBox)
        self.pushButton.setObjectName(u"pushButton")
        self.pushButton.setGeometry(QRect(10, 270, 161, 61))
        font2 = QFont()
        font2.setFamilies([u"Segoe UI"])
        font2.setPointSize(18)
        self.pushButton.setFont(font2)
        self.pushButton.setStyleSheet(u"color: rgb(60, 164, 255);\n"
"\n"
"background-color: rgb(255, 255, 255);")
        self.recode = QPushButton(self.groupBox)
        self.recode.setObjectName(u"recode")
        self.recode.setGeometry(QRect(10, 30, 161, 61))
        font3 = QFont()
        font3.setFamilies([u"Segoe UI Symbol"])
        font3.setPointSize(18)
        font3.setBold(False)
        font3.setItalic(False)
        self.recode.setFont(font3)
        self.recode.setStyleSheet(u"color: rgb(60, 164, 255);\n"
"background-color: rgb(255, 255, 255);")
        self.pushButton_2 = QPushButton(self.groupBox)
        self.pushButton_2.setObjectName(u"pushButton_2")
        self.pushButton_2.setGeometry(QRect(10, 190, 161, 61))
        self.pushButton_2.setFont(font2)
        self.pushButton_2.setStyleSheet(u"color: rgb(60, 164, 255);\n"
"\n"
"background-color: rgb(255, 255, 255);")
        self.save = QPushButton(self.groupBox)
        self.save.setObjectName(u"save")
        self.save.setGeometry(QRect(10, 110, 161, 61))
        self.save.setFont(font2)
        self.save.setStyleSheet(u"color: rgb(60, 164, 255);\n"
"background-color: rgb(255, 255, 255);\n"
"\n"
"\n"
"")
        self.exit = QPushButton(self.groupBox)
        self.exit.setObjectName(u"exit")
        self.exit.setGeometry(QRect(10, 350, 161, 61))
        font4 = QFont()
        font4.setFamilies([u"Segoe UI"])
        font4.setPointSize(18)
        font4.setBold(False)
        font4.setItalic(False)
        self.exit.setFont(font4)
        self.exit.setStyleSheet(u"color: rgb(60, 164, 255);\n"
";\n"
"background-color: rgb(255, 255, 255);")
        self.openrecode = QGroupBox(self.groupBox)
        self.openrecode.setObjectName(u"openrecode")
        self.openrecode.setGeometry(QRect(0, 0, 181, 441))
        self.openrecode.setStyleSheet(u"border-color: rgb(199, 255, 220);\n"
"border-top-color: rgb(138, 255, 146);\n"
"background-color: rgb(185, 246, 255);")
        self.plainTextEdit = QPlainTextEdit(self.openrecode)
        self.plainTextEdit.setObjectName(u"plainTextEdit")
        self.plainTextEdit.setGeometry(QRect(10, 30, 151, 191))
        self.label_4 = QLabel(self.openrecode)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setGeometry(QRect(10, 230, 141, 101))
        self.back = QPushButton(self.openrecode)
        self.back.setObjectName(u"back")
        self.back.setGeometry(QRect(10, 390, 151, 41))
        font5 = QFont()
        font5.setFamilies([u"Tw Cen MT"])
        font5.setPointSize(22)
        self.back.setFont(font5)
        self.back.setStyleSheet(u"background-color: rgb(255, 255, 255);\n"
"color: rgb(224, 99, 255);")
        self.openbb = QPushButton(self.openrecode)
        self.openbb.setObjectName(u"openbb")
        self.openbb.setGeometry(QRect(10, 340, 151, 41))
        self.openbb.setFont(font5)
        self.openbb.setStyleSheet(u"background-color: rgb(255, 255, 255);\n"
"color: rgb(224, 99, 255);")
        self.label_3 = QLabel(self.centralwidget)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setGeometry(QRect(243, 390, 181, 61))
        self.openlb = QLabel(self.centralwidget)
        self.openlb.setObjectName(u"openlb")
        self.openlb.setGeometry(QRect(-260, 190, 371, 281))
        self.groupBox_2 = QGroupBox(self.centralwidget)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.groupBox_2.setGeometry(QRect(280, 120, 621, 441))
        self.label_2 = QLabel(self.groupBox_2)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(10, 50, 601, 381))
        self.label_2.setStyleSheet(u"background-color: rgb(203, 255, 244);")
        self.id = QLineEdit(self.groupBox_2)
        self.id.setObjectName(u"id")
        self.id.setGeometry(QRect(10, 10, 281, 31))
        self.id.setReadOnly(False)
        self.lineEdit = QLineEdit(self.groupBox_2)
        self.lineEdit.setObjectName(u"lineEdit")
        self.lineEdit.setGeometry(QRect(330, 10, 281, 31))
        self.home = QLabel(self.groupBox_2)
        self.home.setObjectName(u"home")
        self.home.setGeometry(QRect(0, 0, 621, 441))
        self.home.setStyleSheet(u"background-color: rgb(203, 255, 244);")
        self.timela = QLabel(self.centralwidget)
        self.timela.setObjectName(u"timela")
        self.timela.setGeometry(QRect(240, 20, 491, 71))
        self.timela.setFont(font1)
        self.timela.setStyleSheet(u"color: rgb(0, 8, 255);\n"
"background-color: rgb(255,255,255);")
        self.timela.setAlignment(Qt.AlignmentFlag.AlignCenter)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 952, 33))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"\u6253\u5361\u7cfb\u7edf", None))
        self.groupBox.setTitle("")
        self.pushButton.setText(QCoreApplication.translate("MainWindow", u"\u8bad\u7ec3", None))
        self.recode.setText(QCoreApplication.translate("MainWindow", u"\u8fdb\u5165\u6253\u5361", None))
        self.pushButton_2.setText(QCoreApplication.translate("MainWindow", u"\u91c7\u96c6", None))
        self.save.setText(QCoreApplication.translate("MainWindow", u"\u5f55\u5165\u4fe1\u606f", None))
        self.exit.setText(QCoreApplication.translate("MainWindow", u"\u9000\u51fa", None))
        self.openrecode.setTitle(QCoreApplication.translate("MainWindow", u"\u6253\u5361\u8bb0\u5f55", None))
        self.label_4.setText("")
        self.back.setText(QCoreApplication.translate("MainWindow", u"\u8fd4\u56de", None))
        self.openbb.setText(QCoreApplication.translate("MainWindow", u"\u6253\u5361", None))
        self.label_3.setText("")
        self.openlb.setText(QCoreApplication.translate("MainWindow", u"TextLabel", None))
        self.groupBox_2.setTitle("")
        self.label_2.setText("")
        self.id.setPlaceholderText(QCoreApplication.translate("MainWindow", u"id", None))
        self.lineEdit.setPlaceholderText(QCoreApplication.translate("MainWindow", u"\u59d3\u540d", None))
        self.home.setText("")
        self.timela.setText("")
    # retranslateUi

