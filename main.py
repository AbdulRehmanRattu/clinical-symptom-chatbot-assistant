import os, sys, shutil
from PyQt5.uic import loadUi
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QAction, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QStandardItemModel, QStandardItem

def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def delete_folder(folder_name):
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)

def copy_file(src_file, dest_file):
    if os.path.exists(src_file):
        shutil.copy(src_file, dest_file)

delete_folder('.reports')
create_folder('.reports')

## For Diagnose X-ray ##

import numpy as np
import cv2
from tensorflow.keras.models import load_model

IMG_SIZE = (150, 150)
img_path = 'resources/images/no_pic.svg'
prediction, prob = '', 0.0

def preprocess_image(img_path):
    image = cv2.imread(img_path)
    image = cv2.resize(image, IMG_SIZE)
    image = image / 255.0
    return np.expand_dims(image, axis=0)
def predict_image_class(model_path, img_path):
    model = load_model(model_path)
    image = preprocess_image(img_path)
    prediction = model.predict(image).ravel()
    prediction_class = (prediction > 0.5).astype(int)
    
    return prediction_class[0], prediction[0]

## For AI Chat ##

# apt-get install tesseract-ocr
import pytesseract
from PIL import Image
from fpdf import FPDF

def image_to_text(image_path):
    """
    Convert image to text using Tesseract OCR.

    :param image_path: Path to the image file
    :return: Extracted text from the image
    """
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        return str(e)

def save_as_pdf(text, output_path):
    """
    Save text content as a PDF file.

    :param text: Text content to be saved
    :param output_path: Path to save the PDF file
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=text.encode('latin-1', 'replace').decode('latin-1'))
    pdf.output(output_path)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
import time

# llm = Ollama(model="moondream")
llm = Ollama(model="phi3")
# llm = Ollama(model="mistral")
# llm = Ollama(model="medllama2")

prompt_template = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Questions:{input}
    """
)

def vector_embedding():
    embeddings = HuggingFaceEmbeddings()
    loader = PyPDFDirectoryLoader(".reports")  # Data Ingestion
    docs = loader.load()  # Document Loading

    if not docs:
      copy_file('resources/empty.pdf', '.reports/empty.pdf')
      docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunk Creation
    final_documents = text_splitter.split_documents(docs)  # Splitting
    vectors = FAISS.from_documents(final_documents, embeddings)  # Vector Hugging Face embeddings
    return vectors

vectors = None

############################################################################

class Home(QMainWindow): # Home
  def __init__(self):
    super(Home, self).__init__()
    loadUi("resources/ui/home.ui", self)
    self.img.setPixmap(QtGui.QPixmap('resources/images/home_side.jpg'))
    #self.label.setText("Rapide Medical AI")
    self.label.adjustSize()

    self.screen1_btn.clicked.connect(self.gotoScreen1)
    self.screen1_btn.adjustSize()
    self.screen2_btn.clicked.connect(self.gotoPreScreen2)
    self.screen2_btn.adjustSize()
    
  def gotoScreen1(self):
      screen1 = Screen1()
      widget.addWidget(screen1)
      widget.setCurrentIndex(widget.currentIndex() + 1)
  
  def gotoPreScreen2(self):
      prescreen2 = PreScreen2()
      widget.addWidget(prescreen2)
      widget.setCurrentIndex(widget.currentIndex() + 1)

class Screen1(QMainWindow): # Diagnose X-ray
  
  def __init__(self):
    super(Screen1, self).__init__()
    loadUi("resources/ui/screen1.ui", self)
    self.button.clicked.connect(self.gotoHome)
    self.choose.clicked.connect(self.browse_img)

    global img_path, prediction, prob
    self.img.setPixmap(QtGui.QPixmap(img_path))
    if prediction != '':
      self.img_name.setText(img_path.split('/')[-1])
      self.result.setText(f"Prediction = {prediction}\nProbability = {prob*100:.2f}%")  
      self.result.adjustSize()

  def browse_img(self):
    fname = QFileDialog.getOpenFileName(self, "Open Image", '', 'Images (*.png *.jpg *.jpeg)')

    if fname[0] == '':
       return

    self.img_name.setText(fname[0].split('/')[-1]) 
    self.img.setPixmap(QtGui.QPixmap(fname[0]))

    global img_path, prediction, prob

    img_path = fname[0]
    prediction_class, prob = predict_image_class('resources/model_95.h5', img_path)
    class_names = {0: "Normal", 1: "Pneumonia"}
    prediction = class_names[prediction_class]
    self.result.setText(f"Prediction = {prediction}\nProbability = {prob*100:.2f}%")  
    self.result.adjustSize()  
  
  def gotoHome(self):
    home = Home()
    widget.addWidget(home)
    widget.setCurrentIndex(widget.currentIndex() + 1)

class PreScreen2(QMainWindow): # Upload Files and create embeddings
  def __init__(self):
    super(PreScreen2, self).__init__()
    loadUi("resources/ui/pre-screen2.ui", self)
    self.label_2.adjustSize()

    self.button.clicked.connect(self.gotoHome)
    self.chat.clicked.connect(self.gotoScreen2)

    self.choose.clicked.connect(self.addFiles)
    self.remove.clicked.connect(self.removeFiles)

    self.files_names = []

  def addFiles(self):
    files, _ = QFileDialog.getOpenFileNames(self, "Open Images or PDFs", '', 'Images and PDF Files (*.png *.jpg *.jpeg *.pdf)')
    
    if files:      
      self.files_names = files

      # Add selected files to the submenu
      for file in files:
        file_action = QAction(file.split('/')[-1], self)
        self.menuUploaded.addAction(file_action)

      # Disable actions in the submenu
      actions = self.menuUploaded.actions()
      for action in actions:
          action.setEnabled(False)
  
      self.remove.setEnabled(True)

      for i in self.files_names:
        if i.endswith('.jpeg') or i.endswith('.jpg') or i.endswith('.png'):
          text = image_to_text(i)
          output_filename = i.split('/')[-1] + '.pdf'
          output_path = os.path.join('', output_filename)
          save_as_pdf(text, '.reports/'+output_path)
        else:
            copy_file(i, '.reports/'+i.split('/')[-1])
    
  def traverse_files(self):
    print(self.files_names)

  def removeFiles(self):
    self.menuUploaded.clear()
    self.remove.setEnabled(False)
    delete_folder('.reports')
    create_folder('.reports')
    
  def gotoHome(self):
      home = Home()
      widget.addWidget(home)
      widget.setCurrentIndex(widget.currentIndex() + 1)
      delete_folder('.reports')
      create_folder('.reports')
  
  def gotoScreen2(self):
      global vectors
      vectors = vector_embedding()
      screen2 = Screen2()
      widget.addWidget(screen2)
      widget.setCurrentIndex(widget.currentIndex() + 1)

class Screen2(QMainWindow): # AI Chat
  def __init__(self):
    super(Screen2, self).__init__()
    loadUi("resources/ui/screen2.ui", self)
    self.label.adjustSize()

    self.button.clicked.connect(self.gotoHome)
    self.ask_btn.clicked.connect(self.send)

    self.input.returnPressed.connect(self.ask_btn.click)

    self.model = QStandardItemModel(self.history)
    self.history.setModel(self.model)
  
  def traverse_messages(self):
    for row in range(self.model.rowCount()):
        item = self.model.item(row)
        print(item.text())

  def send(self):
    message = self.input.text().strip()  # Get the input text and strip leading/trailing spaces
    
    if not message:
        self.showToast("Please enter a message")
        return
    
    item = QStandardItem('Me: '+message)
    self.model.appendRow(item)

    global llm, vectors, prompt_template

    # response_item = QStandardItem("Chatbot: I'm still under development so I'll get back to you when I know the answer!")
    document_chain = create_stuff_documents_chain(llm, prompt_template)
    retriever = vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    start = time.time()
    response = retrieval_chain.invoke({'input': message})
    end = time.time()    
    print("Response time: {:.2f} seconds".format(end - start))
    response_item = response['answer']

    self.model.appendRow(QStandardItem('Syauctus AI: '+response_item+'\n(responded in {:.2f} seconds)'.format(end - start)))
    self.input.clear()  # Clear the input field after sending the message

  def showToast(self, message):
    global screen_geometry, widget_geometry

    toast = ToastNotification(message)
    toast.setGeometry(
      screen_geometry.width() // 3 - widget_geometry.width() // 2,
      screen_geometry.height() // 3 - widget_geometry.height() // 2,
      300, 40
    )
    self.layout().addWidget(toast)
    toast.show()

  def gotoHome(self):
    home = Home()
    widget.addWidget(home)
    widget.setCurrentIndex(widget.currentIndex() + 1)
    delete_folder('.reports')
    create_folder('.reports')
    global vectors
    vectors = None

class ToastNotification(QWidget):
    def __init__(self, message, duration=2000):
        super().__init__()
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("""
            background-color: rgba(0, 0, 0, 0.7); /* Semi-transparent black background */
            color: white;
            border-radius: 5px;
        """)

        self.layout = QVBoxLayout(self)
        self.label = QLabel(message, self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("""
            background-color: rgba(0, 0, 0, 0); /* Transparent background */
            color: white;
        """)
        self.layout.addWidget(self.label)
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.hideToast)
        self.timer.start(duration)

    def hideToast(self):
        self.timer.stop()
        self.setParent(None)

if __name__ == "__main__":

  app = QApplication(sys.argv)
  widget = QtWidgets.QStackedWidget()
  home = Home()
  widget.addWidget(home)
  widget.setFixedWidth(800)
  widget.setFixedHeight(600)
  widget.show()

  # Center the window on the screen
  screen_geometry = app.desktop().availableGeometry()
  widget_geometry = widget.frameGeometry()

  # x, y, width, height
  widget.setGeometry(screen_geometry.width() // 2 - widget_geometry.width() // 2,
                      screen_geometry.height() // 2 - widget_geometry.height() // 2,
                      widget_geometry.width(), widget_geometry.height())

  sys.exit(app.exec_())
