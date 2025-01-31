import sys
import os
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QToolBar, QAction, QFileDialog, \
    QMessageBox, QColorDialog, QSlider, QVBoxLayout, QGridLayout, QLabel, QLineEdit, QPushButton, \
    QHBoxLayout
from PyQt5.QtCore import Qt, QPoint, QSize
from PyQt5.QtGui import QImage, QPainter, QPen, QColor, QPixmap, QIcon, QFont
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import vtk
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
from vtkmodules.vtkIOImage import (
    vtkBMPWriter,
    vtkJPEGWriter,
    vtkPNGWriter,
    vtkPNMWriter,
    vtkPostScriptWriter,
    vtkTIFFWriter
)
from vtkmodules.vtkRenderingCore import vtkWindowToImageFilter
import cv2
import numpy as np
import subprocess
import glob
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QCheckBox, QLineEdit, QPushButton, QGraphicsView, QGraphicsScene, QHBoxLayout, QLabel, QGroupBox, QVBoxLayout, QSplitter
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QPainter, QPen # QColorDialog
from generate_2d_GA import *
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QScrollArea, QLabel
from PyQt5.QtWidgets import QApplication, QMainWindow, QStackedWidget


work_dir = r'F:\exp\topology_optimization\wheels\exp\run_GA'
mini_iteration = 5

def draw_dashed_line(img, start, end, dash_length, color=(0, 0, 0), thickness=1):
    # 计算起点和终点的差值
    dx = end[0] - start[0]
    dy = end[1] - start[1]

    # 计算总长度
    total_length = np.sqrt(dx ** 2 + dy ** 2)

    # 如果总长度小于虚线每段长度，则绘制一条实线
    if total_length <= dash_length:
        cv2.line(img, start, end, color, thickness)
        return

    # 计算虚线中每段和间隔的个数
    dash_section_length = (dash_length + dash_length / 3) # 假设间隔是每段长度的1/3
    num_dashes = int(np.ceil(total_length / dash_section_length))

    # 绘制num_dashes条虚线段
    for i in range(num_dashes):
        x0 = start[0] + dx * dash_section_length * i / total_length
        y0 = start[1] + dy * dash_section_length * i / total_length
        x1 = x0 + dx * dash_length / total_length
        y1 = y0 + dy * dash_length / total_length
        cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)), color, thickness)


class PaintArea(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # 创建一张大小为当前窗口大小的空白图片，填充为透明
        self.image = QImage(self.size(), QImage.Format_ARGB32)
        self.image.fill(Qt.transparent)
        # 加载背景图片
        self.background_image = QImage(self.size(), QImage.Format_ARGB32)
        self.background_image.fill(Qt.white)
        # 初始化画笔状态
        self.drawing = False
        self.brush_size = 3
        self.brush_color = Qt.black
        self.last_point = QPoint()
        self.undo_stack = []
        self.num_folds = 0 # 0表示无阵列
        self.sketch_filename = ''

    def paintEvent(self, event):
        # 当需要重新绘制时调用该函数
        painter = QPainter(self)
        # 绘制背景图片
        painter.drawImage(0, 0, self.background_image)
        # 将当前图片绘制到窗口中
        painter.drawImage(0, 0, self.image)

    def resizeEvent(self, event):
        # 当窗口大小改变时调用该函数
        # 调整当前图片大小为新的窗口大小
        self.image = self.image.scaled(event.size())
        self.background_image = self.background_image.scaled(event.size())

    def mousePressEvent(self, event):
        # 当鼠标按下时调用该函数
        if event.button() == Qt.LeftButton:
            # 将画笔状态设置为正在绘制
            self.drawing = True
            # 记录当前鼠标位置
            self.last_point = event.pos()

    def mouseMoveEvent(self, event):
        # 判断鼠标左键是否按下并且正在绘图
        if (event.buttons() & Qt.LeftButton) and self.drawing:
            # 创建一个画家对象
            painter = QPainter(self.image)
            # 设置画笔的颜色、大小、线条样式、端点样式和连接样式
            painter.setPen(QPen(self.brush_color, self.brush_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            # 画一条线条从上一个点到当前点
            painter.drawLine(self.last_point, event.pos())

            # 将线条沿中心对称阵列
            center = (self.size().width() // 2, self.size().height() // 2)
            for i in range(self.num_folds):
                start = (self.last_point.x(), self.last_point.y())
                end = (event.pos().x(), event.pos().y())

                r = np.sqrt((start[0]-center[0])**2 + (start[1]-center[1])**2)
                theta = np.arctan2(start[1]-center[1], start[0]-center[0]) + 2*np.pi/self.num_folds*i
                start_symmetric = QPoint(center[0] + int(r * np.cos(theta)), center[1] + int(r * np.sin(theta)))

                r = np.sqrt((end[0]-center[0]) ** 2 + (end[1]-center[1]) ** 2)
                theta = np.arctan2(end[1]-center[1], end[0]-center[0]) + 2 * np.pi / self.num_folds*i
                end_symmetric = QPoint(center[0] + int(r * np.cos(theta)), center[1] + int(r * np.sin(theta)))

                painter.drawLine(start_symmetric, end_symmetric)

            # 更新上一个点的位置
            self.last_point = event.pos()

            # 更新视图
            self.update()

    def mouseReleaseEvent(self, event):
        # 判断鼠标是否释放并且正在绘图
        if event.button() == Qt.LeftButton and self.drawing:
            # 停止绘图
            self.drawing = False
            self.undo_stack.append(QImage(self.image))

    def undo_drawing(self):
        if self.undo_stack:
            self.undo_stack.pop()
            if self.undo_stack:
                self.image = QImage(self.undo_stack[-1])
            else:
                self.image.fill(Qt.transparent)
            self.update()

    def set_brush_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.brush_color = color

    def set_brush_size(self, size):
        self.brush_size = size

    def clear_image(self):
        # 用透明色填充图像
        self.image.fill(Qt.transparent)
        # 更新视图
        self.update()
        self.undo_stack.append(QImage(self.image))

    def save_image(self):
        # 跳出文件对话框，让用户选择文件名和保存类型
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Sketch", "",
                                                   "PNG(*.png);;JPEG(*.jpg *.jpeg);;All Files(*.*)")
        # 如果有选择文件名
        if file_name:
            # 保存图像为PNG格式
            if self.image.save(file_name, "PNG"):
                # 显示保存成功的消息框
                QMessageBox.information(self, "Save Sketch", "Sketch saved successfully!")
                img_tmp = cv2.imread(file_name,cv2.IMREAD_UNCHANGED)
                img_tmp = img_tmp[:,:,3]
                cv2.imwrite(file_name, img_tmp)
                QMessageBox.information(self, "Save Sketch", "Transformed to grayscale successfully!")
                self.sketch_filename = file_name

            else:
                # 显示保存失败的消息框
                QMessageBox.warning(self, "Save Image", "Failed to save image!")


class FirstWindow(QWidget):
    def __init__(self, switch_to_second_window):
        super().__init__()
        self.switch_to_second_window = switch_to_second_window
        self.setWindowTitle("设计目标选择与参数设置")
        self.setGeometry(100, 100, 800, 600)

        # 主布局
        self.layout = QHBoxLayout()

        # 左边部分：设计目标选择 + 参数设置 + generate按钮
        self.left_widget = QWidget()
        left_layout = QVBoxLayout(self.left_widget)

        # 设计目标选择区域
        self.goal_group = self.init_goal_group()
        # 添加设计目标选择模块到左边布局
        left_layout.addWidget(self.goal_group)

        # 参数设置区域
        self.param_group = self.init_param_group()
        # 添加参数设置选择模块到左边布局
        left_layout.addWidget(self.param_group)

        # 生成按钮
        self.generate_button = QPushButton("生成设计方案", self)
        self.generate_button.clicked.connect(self.generate_designs)
        left_layout.addWidget(self.generate_button)  # 将按钮添加到左侧布局

        # 查看生成结果按钮
        self.view_button = QPushButton("查看生成结果", self)
        self.view_button.clicked.connect(self.switch_to_second_window)
        left_layout.addWidget(self.view_button)  # 将按钮添加到左侧布局

        # 右边部分：草绘区域
        self.sketch_widget = Sketch()

        # 使用QSplitter让左右区域可调整
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.left_widget)  # 将左边的widget添加到splitter
        splitter.addWidget(self.sketch_widget)  # 将草绘区域添加到splitter
        splitter.setStretchFactor(1.2, 2)  # 调整右侧草绘区域的占比（2倍）

        self.layout.addWidget(splitter)

        self.setLayout(self.layout)


    # 设计目标选择区域
    def init_goal_group(self):

        self.goal_group = QGroupBox("设计目标选择")
        self.goal_layout = QVBoxLayout()

        self.first_order_modal_frequency = QCheckBox("一阶模态振频")
        self.radial_stress_checkbox = QCheckBox("径向载荷下的最大应力")
        self.radial_displacement_checkbox = QCheckBox("径向载荷下的最大位移")
        self.radial_stress_map_checkbox = QCheckBox("径向载荷下的应力分布")
        self.bending_stress_checkbox = QCheckBox("弯曲载荷下的最大应力")
        self.bending_displacement_checkbox = QCheckBox("弯曲载荷下的最大位移")
        self.bending_stress_map_checkbox = QCheckBox("弯曲载荷下的应力分布")

        self.goal_layout.addWidget(self.first_order_modal_frequency)
        self.goal_layout.addWidget(self.radial_stress_checkbox)
        self.goal_layout.addWidget(self.radial_displacement_checkbox)
        self.goal_layout.addWidget(self.radial_stress_map_checkbox)
        self.goal_layout.addWidget(self.bending_stress_checkbox)
        self.goal_layout.addWidget(self.bending_displacement_checkbox)
        self.goal_layout.addWidget(self.bending_stress_map_checkbox)

        self.goal_group.setLayout(self.goal_layout)
        return  self.goal_group

    # 参数设置区域
    def init_param_group(self):

        self.param_group = QGroupBox("参数设置")

        self.param_layout = QVBoxLayout()  # 创建一个垂直布局

        self.dvf_label = QLabel("目标体积率")
        self.dvf_input = QLineEdit(self)
        self.dvf_input.setPlaceholderText("0.35")
        self.dvf_input.setStyleSheet("QLineEdit { border: 1px solid gray; }")  # 设置输入框的样式
        self.dvf_layout = QHBoxLayout()
        self.dvf_layout.addWidget(self.dvf_label)
        self.dvf_layout.addWidget(self.dvf_input)

        self.iteration_label = QLabel("迭代次数")
        self.iteration_input = QLineEdit(self)
        self.iteration_input.setPlaceholderText("30")
        self.iteration_input.setStyleSheet("QLineEdit { border: 1px solid gray; }")  # 设置输入框的样式
        self.iteration_layout = QHBoxLayout()
        self.iteration_layout.addWidget(self.iteration_label)
        self.iteration_layout.addWidget(self.iteration_input)

        self.resolution_fea_label = QLabel("有限元网格数")
        self.resolution_fea_input = QLineEdit(self)
        self.resolution_fea_input.setPlaceholderText("180")
        self.resolution_fea_input.setStyleSheet("QLineEdit { border: 1px solid gray; }")  # 设置输入框的样式
        self.resolution_fea_layout = QHBoxLayout()
        self.resolution_fea_layout.addWidget(self.resolution_fea_label)
        self.resolution_fea_layout.addWidget(self.resolution_fea_input)

        self.batch_size_label = QLabel("生成方案数量")
        self.batch_size_input = QLineEdit(self)
        self.batch_size_input.setPlaceholderText("100")
        self.batch_size_input.setStyleSheet("QLineEdit { border: 1px solid gray; }")  # 设置输入框的样式
        self.batch_size_layout = QHBoxLayout()
        self.batch_size_layout.addWidget(self.batch_size_label)
        self.batch_size_layout.addWidget(self.batch_size_input)

        self.lmin_label = QLabel("形态特征细粒度")
        self.lmin_input = QLineEdit(self)
        self.lmin_input.setPlaceholderText("8")
        self.lmin_input.setStyleSheet("QLineEdit { border: 1px solid gray; }")  # 设置输入框的样式
        self.lmin_layout = QHBoxLayout()
        self.lmin_layout.addWidget(self.lmin_label)
        self.lmin_layout.addWidget(self.lmin_input)

        self.param_layout.addLayout(self.dvf_layout)
        self.param_layout.addLayout(self.iteration_layout)
        self.param_layout.addLayout(self.resolution_fea_layout)
        self.param_layout.addLayout(self.batch_size_layout)
        self.param_layout.addLayout(self.lmin_layout)

        self.param_group.setLayout(self.param_layout)
        return self.param_group


    def generate_designs(self):
        # 获取选择的目标
        selected_goals = []
        if self.first_order_modal_frequency.isChecked():
            selected_goals.append("一阶模态振频")
        if self.radial_stress_checkbox.isChecked():
            selected_goals.append("径向载荷下的最大应力")
        if self.radial_displacement_checkbox.isChecked():
            selected_goals.append("径向载荷下的最大位移")
        if self.radial_stress_map_checkbox.isChecked():
            selected_goals.append("径向载荷下的应力分布")
        if self.bending_stress_checkbox.isChecked():
            selected_goals.append("弯曲载荷下的最大应力")
        if self.bending_displacement_checkbox.isChecked():
            selected_goals.append("弯曲载荷下的最大位移")
        if self.bending_stress_map_checkbox.isChecked():
            selected_goals.append("弯曲载荷下的应力分布")


        # 调用函数，生成设计方案
        GA(
            batch_size=int(self.batch_size_input.text()),
            iteration=int(self.iteration_input.text()),
            mini_iteration=mini_iteration,
            resolution_fea=int(self.resolution_fea_input.text()),
            resolution_image=int(self.resolution_fea_input.text()),
            dvf=float(self.dvf_input.text()),
            numFolds=int(self.sketch_widget.textbox.text()),
            ref_img=self.sketch_widget.paint_area.sketch_filename,
            lmin=float(self.lmin_input.text()),
            output_dir=os.path.join(work_dir, 'samples', '2d')
        )

    def select_color(self):
        # 选择颜色
        color = QColorDialog.getColor()
        if color.isValid():
            self.paint_area.set_pen_color(color)

    def update_pen_width(self):
        try:
            width = int(self.pen_width_input.text())
            self.paint_area.set_pen_width(width)
        except ValueError:
            pass

    def update_num_folds(self):
        try:
            num_folds = int(self.num_folds_input.text())
            self.paint_area.set_num_folds(num_folds)
        except ValueError:
            pass


class Sketch(QWidget):
    def __init__(self):
        super(Sketch, self).__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("2D Paint")
        self.setGeometry(400, 50, 900, 960)

        layout = QVBoxLayout(self)
        # self.setCentralWidget(QWidget()) # 创建中心部件
        # layout = QVBoxLayout(self.centralWidget()) # 创建中心部件的布局管理器

        # 创建画布
        self.paint_area = PaintArea()

        # 创建自定义的“工具栏”
        self.custom_toolbar = QWidget()
        self.custom_toolbar.setStyleSheet("background-color: lightgray;")  # 设置背景色以区分
        self.custom_toolbar.setFixedHeight(50)  # 设置工具栏的高度（固定高度40）
        toolbar_layout = QHBoxLayout(self.custom_toolbar)

        # 设置文字字体大小
        font = QFont()
        font.setPointSize(12)

        color_button = QPushButton(QIcon("icons/palette.png"), " Color", self.custom_toolbar)
        color_button.setIconSize(QSize(32, 32))  # 设置图标大小
        color_button.setFont(font)  # 设置文字字体
        color_button.setFixedSize(100, 40)  # 设置按钮大小，宽度为150，高度为50
        color_button.setToolTip('Color')  # 设置工具提示
        color_button.clicked.connect(self.paint_area.set_brush_color)
        toolbar_layout.addWidget(color_button)

        clear_button = QPushButton(QIcon("icons/eraser.png"), " Clear", self.custom_toolbar)
        clear_button.setIconSize(QSize(32, 32))  # 设置图标大小
        clear_button.setFont(font)  # 设置文字字体
        clear_button.setFixedSize(100, 40)  # 设置按钮大小，宽度为150，高度为50
        clear_button.setToolTip('Clear')  # 设置工具提示
        clear_button.clicked.connect(self.paint_area.clear_image)
        toolbar_layout.addWidget(clear_button)

        undo_button = QPushButton(QIcon("icons/undo.png"), " Undo", self.custom_toolbar)
        undo_button.setIconSize(QSize(32, 32))  # 设置图标大小
        undo_button.setFont(font)  # 设置文字字体
        undo_button.setFixedSize(100, 40)  # 设置按钮大小，宽度为150，高度为50
        undo_button.setToolTip('Undo')  # 设置工具提示
        undo_button.clicked.connect(self.paint_area.undo_drawing)
        toolbar_layout.addWidget(undo_button)

        save_button = QPushButton(QIcon("icons/save.png"), " Save", self.custom_toolbar)
        save_button.setIconSize(QSize(32, 32))  # 设置图标大小
        save_button.setFont(font)  # 设置文字字体
        save_button.setFixedSize(100, 40)  # 设置按钮大小，宽度为150，高度为50
        save_button.setToolTip('Save')  # 设置工具提示
        save_button.clicked.connect(self.paint_area.save_image)
        toolbar_layout.addWidget(save_button)

        # 输入num_split
        num_split_input = QWidget()
        text_layout = QHBoxLayout(num_split_input)
        text_hint = QLabel('Folds: ')
        text_hint.setFont(font)  # 设置文字字体
        text_hint.setFixedHeight(40)
        text_hint.setFixedWidth(45)
        text_layout.addWidget(text_hint)

        self.textbox = QLineEdit()
        self.textbox.setPlaceholderText("6")
        self.textbox.setFixedHeight(40)  # 设置固定高度
        self.textbox.setFixedWidth(20)  # 设置固定宽度
        self.textbox.setFont(font)  # 设置文字字体
        self.textbox.setStyleSheet("QLineEdit { border: none; }") # 设置边框透明
        text_layout.addWidget(self.textbox)

        show_canvas_button = QPushButton('Show canvas', self.custom_toolbar)
        show_canvas_button.setFont(font)  # 设置文字字体
        show_canvas_button.setFixedSize(110, 40)  # 设置按钮大小
        show_canvas_button.clicked.connect(self.show_canvas)

        text_layout.addWidget(show_canvas_button)
        text_layout.setAlignment(Qt.AlignLeft)
        text_layout.setContentsMargins(5, 0, 0, 0)  # 设置边距

        toolbar_layout.addWidget(num_split_input)
        toolbar_layout.setAlignment(Qt.AlignLeft)
        toolbar_layout.setContentsMargins(5, 0, 0, 0)  # 设置边距

        # 将自定义的“工具栏”添加到垂直布局中
        layout.addWidget(self.custom_toolbar)

        # 将画布添加到垂直布局中
        layout.addWidget(self.paint_area)


    def show_canvas(self):

        # 设置画布为尽可能大的正方形
        resolution = min(self.paint_area.size().width(), self.paint_area.size().height())
        resolution = (resolution // 2) * 2
        self.paint_area.setFixedSize(resolution, resolution)

        try:
            num_folds = int(self.textbox.text())
            assert (num_folds > 2) and (num_folds < 20)
        except:
            QMessageBox.information(self, "Warning", "Invalid number of folds")
            return

        if self.create_sketch_canvas(resolution = resolution,
                                   center = (resolution // 2, resolution // 2),
                                   num_split = num_folds,
                                   thickness = 2,
                                   line_thickness = 1,
                                   save_filename='sketch_canvas.png'):

            self.paint_area.background_image = QImage('sketch_canvas.png')
            self.paint_area.num_folds = num_folds
            self.paint_area.update()


    def create_sketch_canvas(self, resolution, center, num_split, thickness, line_thickness, save_filename):
        import cv2
        import numpy as np

        # 定义绘制参数
        angle = 0  # 椭圆不旋转
        startAngle = 0  # 圆弧起始角度
        endAngle = 360  # 圆弧结束角度
        color = (0, 0, 0)  # 黑色

        # 创建一个白色的图像
        img = 255 * np.ones((resolution, resolution, 3), np.uint8)

        # 绘制rim上线条
        num_lines = 30
        for i in range(num_lines):
            x0 = 0;
            y0 = resolution - int(resolution / num_lines) * i
            x1 = int(resolution / num_lines) * i;
            y1 = resolution
            cv2.line(img, (x0, y0), (x1, y1), color, line_thickness)
        for i in range(num_lines):
            x0 = int(resolution / num_lines) * i;
            y0 = 0
            x1 = resolution;
            y1 = resolution - int(resolution / num_lines) * i
            cv2.line(img, (x0, y0), (x1, y1), color, line_thickness)

        r = int((resolution // 2) * 0.95)
        rim_outer = (r, r)  # 椭圆的长轴和短轴长度的一半
        r = int((resolution // 2) * 0.84)
        rim_inner = (r, r)
        r = int((resolution // 2) * 0.35)
        hub_outer = (r, r)
        r = int((resolution // 2) * 0.22)
        hub_inner = (r, r)

        # 绘制rim和hub
        mask = np.zeros((resolution, resolution), dtype=np.bool)  # 创建一个与图像大小相同的掩码数组，初始化为0（表示不可见）
        yy, xx = np.ogrid[:mask.shape[0], :mask.shape[1]]
        dist_sq = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
        mask = (dist_sq >= rim_inner[0] ** 2) & (dist_sq <= rim_outer[0] ** 2)  # 使用NumPy的广播和条件表达式来设置圆环区域的掩码为True
        img = np.array(img, dtype=np.uint8)
        img[~mask] = np.array([255, 255, 255])

        cv2.ellipse(img, center, rim_outer, angle, startAngle, endAngle, color, thickness)
        cv2.ellipse(img, center, rim_inner, angle, startAngle, endAngle, color, thickness)
        cv2.ellipse(img, center, hub_outer, angle, startAngle, endAngle, color, thickness)
        cv2.ellipse(img, center, hub_inner, angle, startAngle, endAngle, color, thickness)

        # 绘制安装孔
        r_inner = int((resolution // 2) * 0.041)
        r_outer = int((resolution // 2) * 0.027)
        r_pcd = int((hub_inner[0] + hub_outer[0]) / 2)
        for i in range(num_split):
            y = center[1] + r_pcd * np.sin(np.deg2rad((180. / num_split) * (2 * i + 1)))
            x = center[0] + r_pcd * np.cos(np.deg2rad((180. / num_split) * (2 * i + 1)))
            center_hole = (int(x), int(y))
            cv2.ellipse(img, center_hole, (r_inner, r_inner), angle, startAngle, endAngle, color, thickness)
            cv2.ellipse(img, center_hole, (r_outer, r_outer), angle, startAngle, endAngle, color, thickness)

        # 绘制中心对称线
        for i in range(num_split):
            y = center[1] + int((resolution // 2) * 0.985) * np.sin(np.deg2rad(360. / num_split * i -90 + 360. / num_split / 2))
            x = center[0] + int((resolution // 2) * 0.985) * np.cos(np.deg2rad(360. / num_split * i -90 + 360. / num_split / 2))
            end_point = (x, y)
            dash_length = int((resolution // 2) * 0.05)
            draw_dashed_line(img, center, end_point, dash_length, color=(0, 0, 0), thickness=line_thickness)

        # 保存图像为PNG文件
        try:
            cv2.imwrite(save_filename, img)
            return True
        except:
            return False


class SecondWindow(QWidget):
    def __init__(self, switch_to_third_window, columns=3):
        super(SecondWindow, self).__init__()
        self.switch_to_third_window = switch_to_third_window
        self.columns = columns  # 列数
        self.show_data = {}
        self.init_ui()
        self.

    def read_imgs(self, dir):
        self.show_data = {}
        for img in glob.glob(os.path.join(dir, '*.png')):
            self.show_data[img] = {}
            self.show_data[img]['comp'] = float(img.split('comp_')[1].split('_vol')[0])
            self.show_data[img]['vol'] = float(img.split('vol_')[1].strip('.png'))

    def init_ui(self):
        self.setWindowTitle("生成的设计方案")
        self.setGeometry(400, 50, 900, 600)

        main_layout = QVBoxLayout(self)

        # 创建滚动区域来展示生成的图像
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_widget = QWidget(self.scroll_area)
        self.scroll_area.setWidget(self.scroll_widget)

        # 使用 QGridLayout 将图片网格化排列
        self.grid_layout = QGridLayout(self.scroll_widget)

        # 读取生成的图像路径和性能指标
        base_dir = os.path.join(work_dir, 'samples', '2d')
        dir_name = sorted(os.listdir(base_dir), key=lambda x:int(x.strip('iteration_')))[-1]
        dir = os.path.join(base_dir, dir_name)
        self.read_imgs(dir=dir)

        # 网格布局：添加图像和性能指标
        for index, img_path in enumerate(self.show_data):
            # 计算行列位置
            row = index // self.columns
            col = index % self.columns

            # 图片展示
            pixmap = QPixmap(img_path)
            img_label = QLabel(self)
            img_label.setPixmap(pixmap.scaled(150, 150, Qt.KeepAspectRatio))

            # 性能指标展示
            comp_vol = 'compliance %.3f\nvolume %.3f' % (self.show_data[img_path]['comp'], self.show_data[img_path]['vol'])
            performance_label = QLabel(comp_vol, self)

            # Detail 按钮
            detail_button = QPushButton("Detail", self)
            name = os.path.basename(img_path).split('.png')[0]
            ply_path = os.path.join(work_dir, 'samples','ply', name+'.ply')
            detail_button.clicked.connect(lambda: self.switch_to_third_window(ply_path))

            # 将图片和性能指标加入网格布局
            self.grid_layout.addWidget(img_label, row * 2, col)  # 图片位置
            self.grid_layout.addWidget(performance_label, row * 2 + 1, col)  # 指标位置
            self.grid_layout.addWidget(detail_button, row * 3 + 1, col)  # 指标位置

        main_layout.addWidget(self.scroll_area)
        self.setLayout(main_layout)


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        # 初始化主窗口
        self.setWindowTitle("设计方案生成")
        self.setGeometry(400, 50, 900, 600)

        # 创建QStackedWidget来管理多个页面
        self.stacked_widget = QStackedWidget(self)
        self.setCentralWidget(self.stacked_widget)

        # 创建并添加第一个\第二个\第三个界面
        self.first_window = FirstWindow(self.switch_to_second_window)  # 第一个界面
        self.second_window = SecondWindow(self.switch_to_third_window)  # 第二个界面
        self.third_window = ThirdWindow(ply_filename=)  # 第三个界面

        # 将界面添加到QStackedWidget
        self.stacked_widget.addWidget(self.first_window)
        self.stacked_widget.addWidget(self.second_window)

        # 设置初始显示第一个界面
        self.stacked_widget.setCurrentWidget(self.first_window)

    def switch_to_second_window(self):
        """切换到第二个界面"""
        self.stacked_widget.setCurrentWidget(self.second_window)

    def switch_to_third_window(self):
        """切换到第三个界面"""
        self.stacked_widget.setCurrentWidget(self.third_window)


class MyRenderWindow(QVTKRenderWindowInteractor):
    def __init__(self, parent=None):
        super(MyRenderWindow, self).__init__(parent)

    def closeEvent(self, event):
        self.GetRenderWindow().Finalize()
        self.GetRenderWindow().Delete()
        event.accept()


class ThirdWindow(QWidget):
    def __init__(self, ply_filename):
        super(ThirdWindow, self).__init__()
        self.ply_filename = ply_filename
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("3D Model Viewer")
        self.setGeometry(200, 50, 800, 600)

        # 使用 QVTKRenderWindowInteractor 创建 VTK 渲染窗口
        self.vtk_widget = MyRenderWindow(self)  # 使用自定义的事件处理器

        # 创建 VTK 渲染器
        self.renderer = vtk.vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)

        # 读取 PLY 文件
        self.reader = vtk.vtkPLYReader()
        self.reader.SetFileName(self.ply_filename)  # 设置你的 .ply 文件路径

        # 创建映射器和演员
        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.SetInputConnection(self.reader.GetOutputPort())

        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.mapper)

        # 添加演员到渲染器
        self.renderer.AddActor(self.actor)

        # 设置背景色为白色
        self.renderer.SetBackground(1, 1, 1)

        # 初始化 VTK 交互器
        self.vtk_widget.GetRenderWindow().Render()  # 强制更新画面
        self.vtk_widget.GetRenderWindow().GetInteractor().Initialize()


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == '__main__':

    main()
