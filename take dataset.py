import cv2
import mediapipe as mp
import csv
import time
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

csv_filename = "hand_pose_dataset.csv"

if not os.path.exists(csv_filename):
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ['label'] + [f'x{i},y{i},z{i}' for i in range(21)] 
        writer.writerow(header)

def get_existing_labels():
    labels = set()
    with open(csv_filename, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  
        for row in reader:
            labels.add(row[0])
    return list(labels)

def save_landmark_data(label, landmarks):
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        row = [label] + [f"{lm.x},{lm.y},{lm.z}" for lm in landmarks]
        writer.writerow(row)

def count_data_per_label(label):
    with open(csv_filename, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  
        count = sum(1 for row in reader if row[0] == label)
    return count

collecting_data = False
label = None
last_saved_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if collecting_data and label:
                current_time = time.time()
                data_count = count_data_per_label(label)

                if data_count >= 300:
                    print(f"Data untuk label '{label}' telah mencapai 300. Pengambilan dihentikan.")
                    collecting_data = False
                    label = None
                elif current_time - last_saved_time >= 1:  # Setiap 3 detik
                    save_landmark_data(label, hand_landmarks.landmark)
                    print(f"Data untuk label '{label}' disimpan! ({data_count + 1}/300)")
                    last_saved_time = current_time

    cv2.putText(frame, "Tekan SPASI untuk menambah/ambil data label", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if collecting_data:
        cv2.putText(frame, f"Label aktif: {label}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, "Tekan ENTER untuk berhenti menyimpan data", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Hand Pose Estimation", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '): 
        print("Pilih opsi:")
        print("1. Tambah data pada label yang ada")
        print("2. Masukkan label baru")
        choice = input("Masukkan pilihan (1/2): ")

        if choice == '1':
            existing_labels = get_existing_labels()
            if not existing_labels:
                print("Belum ada label yang tersedia. Masukkan label baru.")
                continue
            print("Label yang tersedia:")
            for i, lbl in enumerate(existing_labels):
                print(f"{i + 1}. {lbl}")
            label_choice = int(input("Pilih label (masukkan nomor): ")) - 1
            label = existing_labels[label_choice]
        elif choice == '2':
            label = input("Masukkan label baru: ")
        else:
            print("Pilihan tidak valid.")
            continue

        data_count = count_data_per_label(label)
        if data_count >= 300:
            print(f"Data untuk label '{label}' telah mencapai 300. Tidak dapat menambahkan lebih banyak data.")
            label = None
        else:
            collecting_data = True
            last_saved_time = time.time()  

    if key == 13:  
        print(f"Pengambilan data untuk label '{label}' dihentikan.")
        collecting_data = False
        label = None

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
