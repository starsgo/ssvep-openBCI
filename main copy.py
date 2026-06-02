import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class SimpleWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt 简单界面")
        self.resize(420, 260)

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("请输入你的名字")

        self.theme_select = QComboBox()
        self.theme_select.addItems(["浅色", "深色"])
        self.theme_select.currentTextChanged.connect(self.apply_theme)

        self.message_label = QLabel("欢迎使用 PyQt 界面")
        self.message_label.setAlignment(Qt.AlignCenter)
        self.message_label.setObjectName("messageLabel")

        confirm_button = QPushButton("生成问候")
        confirm_button.clicked.connect(self.update_message)

        clear_button = QPushButton("清空")
        clear_button.clicked.connect(self.clear_form)

        form_layout = QVBoxLayout()
        form_layout.addWidget(QLabel("名字"))
        form_layout.addWidget(self.name_input)
        form_layout.addWidget(QLabel("主题"))
        form_layout.addWidget(self.theme_select)

        button_layout = QHBoxLayout()
        button_layout.addWidget(confirm_button)
        button_layout.addWidget(clear_button)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(28, 24, 28, 24)
        main_layout.setSpacing(14)
        main_layout.addLayout(form_layout)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.message_label)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.apply_theme(self.theme_select.currentText())

    def update_message(self):
        name = self.name_input.text().strip()
        if name:
            self.message_label.setText(f"你好，{name}！")
        else:
            self.message_label.setText("你好，请先输入名字。")

    def clear_form(self):
        self.name_input.clear()
        self.message_label.setText("欢迎使用 PyQt 界面")

    def apply_theme(self, theme):
        if theme == "深色":
            self.setStyleSheet(
                """
                QWidget {
                    background: #202124;
                    color: #f1f3f4;
                    font-size: 14px;
                }
                QLineEdit, QComboBox {
                    background: #303134;
                    border: 1px solid #5f6368;
                    border-radius: 6px;
                    padding: 8px;
                }
                QPushButton {
                    background: #8ab4f8;
                    color: #202124;
                    border: none;
                    border-radius: 6px;
                    padding: 9px 14px;
                }
                QPushButton:hover {
                    background: #aecbfa;
                }
                QLabel#messageLabel {
                    background: #303134;
                    border-radius: 8px;
                    padding: 18px;
                    font-size: 18px;
                    font-weight: 600;
                }
                """
            )
            return

        self.setStyleSheet(
            """
            QWidget {
                background: #f7f8fa;
                color: #202124;
                font-size: 14px;
            }
            QLineEdit, QComboBox {
                background: white;
                border: 1px solid #d0d5dd;
                border-radius: 6px;
                padding: 8px;
            }
            QPushButton {
                background: #2563eb;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 9px 14px;
            }
            QPushButton:hover {
                background: #1d4ed8;
            }
            QLabel#messageLabel {
                background: white;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                padding: 18px;
                font-size: 18px;
                font-weight: 600;
            }
            """
        )


def main():
    app = QApplication(sys.argv)
    window = SimpleWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
