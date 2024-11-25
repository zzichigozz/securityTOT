import streamlit as st
import cv2
import pandas as pd
from deepface import DeepFace
import os
from datetime import datetime
from fpdf import FPDF  # Убедитесь, что fpdf установлена

# Папки для данных
DATA_DIR = "data"
PHOTOS_DIR = os.path.join(DATA_DIR, "photos")
EMPLOYEES_FILE = os.path.join(DATA_DIR, "employees.csv")
VISITS_FILE = os.path.join(DATA_DIR, "visits.csv")

# Проверка папок и файлов
os.makedirs(PHOTOS_DIR, exist_ok=True)
if not os.path.exists(EMPLOYEES_FILE):
    pd.DataFrame(columns=["fio", "position", "photo"]).to_csv(EMPLOYEES_FILE, index=False)
if not os.path.exists(VISITS_FILE):
    pd.DataFrame(columns=["fio", "time"]).to_csv(VISITS_FILE, index=False)


# Функции
def load_employees():
    return pd.read_csv(EMPLOYEES_FILE)


def load_visits():
    return pd.read_csv(VISITS_FILE)


def log_visit(fio):
    visits = load_visits()
    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_visit = pd.DataFrame({"fio": [fio], "time": [time]})
    visits = pd.concat([visits, new_visit], ignore_index=True)
    visits.to_csv(VISITS_FILE, index=False)


# Функция для генерации PDF
def generate_pdf(visits, output_path):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Заголовок отчета
    pdf.set_font("Arial", size=12, style='B')
    pdf.cell(200, 10, "Отчет о посещениях сотрудников", ln=True, align="C")

    # Таблица посещений
    pdf.ln(10)  # Отступ
    pdf.set_font("Arial", size=10)
    pdf.cell(95, 10, "ФИО", border=1, align='C')
    pdf.cell(95, 10, "Время", border=1, align='C')
    pdf.ln(10)

    for index, row in visits.iterrows():
        pdf.cell(95, 10, row["fio"], border=1, align='C')
        pdf.cell(95, 10, row["time"], border=1, align='C')
        pdf.ln(10)

    pdf.output(output_path)


# Интерфейс приложения
st.title("Система распознавания сотрудников")
mode = st.sidebar.selectbox("Режим работы", ["Распознавание", "Админка", "Отчет"])

if mode == "Распознавание":
    st.subheader("Распознавание сотрудников")
    employees = load_employees()

    # Загружаем фото сотрудников и привязываем их к ФИО и должности
    known_employees = []
    for _, row in employees.iterrows():
        photo_path = row["photo"]
        if os.path.exists(photo_path):
            known_employees.append({
                "fio": row["fio"],
                "position": row["position"],
                "photo": photo_path
            })

    # Подключение к камере
    video_capture = cv2.VideoCapture(0)
    stframe = st.empty()

    recognized_fios = set()  # Для отслеживания сотрудников, которых уже распознали

    while True:
        ret, frame = video_capture.read()
        if not ret:
            st.warning("Камера не найдена!")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Используем OpenCV для определения лиц в кадре
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

        recognized = False
        for (x, y, w, h) in faces:
            # Обрезаем лицо
            face_region = rgb_frame[y:y + h, x:x + w]
            face = cv2.cvtColor(face_region, cv2.COLOR_RGB2BGR)

            # Проверяем распознавание лица для каждого сотрудника
            for employee in known_employees:
                try:
                    result = DeepFace.verify(face, employee["photo"], enforce_detection=False)
                    if result["verified"]:
                        fio = employee["fio"]
                        position = employee["position"]

                        # Рисуем квадрат вокруг лица
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, f"{fio}, {position}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                    (0, 255, 0), 2)
                        recognized = True

                        if fio not in recognized_fios:  # Если сотрудник еще не зарегистрирован
                            log_visit(fio)  # Регистрируем время появления сотрудника
                            recognized_fios.add(fio)
                        break
                except Exception as e:
                    st.error(f"Ошибка распознавания: {e}")

        # Если никто не найден в кадре, не отображаем информацию
        if not recognized:
            cv2.putText(frame, "Нет сотрудников в кадре", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        stframe.image(frame, channels="BGR")

elif mode == "Админка":
    st.subheader("Управление сотрудниками")

    # Загрузка данных сотрудников
    employees = load_employees()

    # Форма для добавления нового сотрудника
    with st.form(key="add_employee_form"):
        fio = st.text_input("ФИО сотрудника")
        position = st.text_input("Должность")
        photo = st.file_uploader("Фото сотрудника", type=["jpg", "jpeg", "png"])
        submit_button = st.form_submit_button("Добавить сотрудника")

        if submit_button:
            if fio and position and photo:
                photo_path = os.path.join(PHOTOS_DIR, f"{fio.replace(' ', '_')}.jpg")
                with open(photo_path, "wb") as f:
                    f.write(photo.getbuffer())

                new_employee = pd.DataFrame({"fio": [fio], "position": [position], "photo": [photo_path]})
                employees = pd.concat([employees, new_employee], ignore_index=True)
                employees.to_csv(EMPLOYEES_FILE, index=False)
                st.success(f"Сотрудник {fio} добавлен.")

    # Таблица сотрудников
    if not employees.empty:
        st.subheader("Сотрудники в системе")
        st.write(employees)

        # Кнопка для удаления сотрудника
        with st.form(key="delete_employee_form"):
            delete_fio = st.selectbox("Выберите сотрудника для удаления", employees["fio"].tolist())
            delete_button = st.form_submit_button("Удалить сотрудника")

            if delete_button:
                employees = employees[employees["fio"] != delete_fio]
                employees.to_csv(EMPLOYEES_FILE, index=False)
                st.success(f"Сотрудник {delete_fio} удален.")

elif mode == "Отчет":
    st.subheader("Отчет о посещениях сотрудников")
    visits = load_visits()

    if visits.empty:
        st.warning("Нет данных о посещениях.")
    else:
        st.subheader("Данные о посещениях")
        st.write(visits)

        # Кнопка для скачивания отчета
        if st.button("Скачать PDF с отчетом"):
            output_path = os.path.join(DATA_DIR, "visits_report.pdf")

            # Генерация PDF
            generate_pdf(visits, output_path)

            # Выгрузка PDF
            with open(output_path, "rb") as f:
                st.download_button("Скачать PDF", data=f, file_name="visits_report.pdf")

