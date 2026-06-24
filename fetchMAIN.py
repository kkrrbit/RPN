import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np
import os
import threading
import time


class HoleDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("СМЗ - Система поиска отверстий")
        self.root.geometry("1100x950")

        # Переменные для камеры
        self.camera = None
        self.is_running = False
        self.current_image = None
        self.processed_image = None
        self.snapshot_window = None

        # Параметры для HoughCircles (ваши подобранные значения)
        self.dp = 0.7
        self.minDist = 50
        self.param1 = 30
        self.param2 = 32
        self.minRadius = 20
        self.maxRadius = 40

        # Параметры автонастройки
        self.auto_tune_enabled = False
        self.target_holes = 43
        self.is_tuning = False

        self.create_widgets()

    def create_widgets(self):
        # Левая панель - изображение
        left_frame = ttk.LabelFrame(self.root, text="Изображение", padding=5)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.image_label = ttk.Label(left_frame, text="Здесь будет изображение",
                                     background="black", foreground="white")
        self.image_label.pack(fill=tk.BOTH, expand=True)

        # Правая панель - управление
        right_frame = ttk.LabelFrame(self.root, text="Управление", padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

        # Режимы работы
        ttk.Label(right_frame, text="Режим работы:", font=("Arial", 12, "bold")).pack(pady=5)

        self.mode_var = tk.StringVar(value="photo")
        ttk.Radiobutton(right_frame, text="Анализ фото", variable=self.mode_var,
                        value="photo", command=self.switch_mode).pack(anchor="w", pady=2)
        ttk.Radiobutton(right_frame, text="Анализ с камеры USB", variable=self.mode_var,
                        value="camera", command=self.switch_mode).pack(anchor="w", pady=2)

        ttk.Separator(right_frame, orient='horizontal').pack(fill='x', pady=10)

        # Кнопки для режима фото
        self.photo_frame = ttk.Frame(right_frame)
        ttk.Button(self.photo_frame, text="Загрузить фото",
                   command=self.load_image).pack(fill='x', pady=2)
        ttk.Button(self.photo_frame, text="Найти отверстия",
                   command=self.process_image).pack(fill='x', pady=2)

        # Кнопки для режима камеры
        self.camera_frame = ttk.Frame(right_frame)
        self.cam_btn = ttk.Button(self.camera_frame, text="Запустить камеру",
                                  command=self.toggle_camera)
        self.cam_btn.pack(fill='x', pady=2)
        ttk.Button(self.camera_frame, text="Сделать снимок и найти отверстия",
                   command=self.capture_and_process).pack(fill='x', pady=2)

        # Показываем кнопки для фото, скрываем для камеры
        self.camera_frame.pack_forget()
        self.photo_frame.pack(fill='x', pady=5)

        ttk.Separator(right_frame, orient='horizontal').pack(fill='x', pady=10)

        # Координаты найденных отверстий
        ttk.Label(right_frame, text="Найденные отверстия:", font=("Arial", 12, "bold")).pack(pady=5)

        self.coords_text = tk.Text(right_frame, height=10, width=35)
        self.coords_text.pack(pady=5, fill='x')

        # Кнопка сохранения результата
        ttk.Button(right_frame, text="Сохранить результат",
                   command=self.save_result).pack(fill='x', pady=5)

        # Статус
        self.status_label = ttk.Label(right_frame, text="Статус: готов", foreground="green")
        self.status_label.pack(pady=10)

        # Параметры поиска с подсказками
        params_frame = ttk.LabelFrame(right_frame, text="Настройки поиска", padding=5)
        params_frame.pack(fill='x', pady=10)

        # Строка 1
        ttk.Label(params_frame, text="dp (0.5-1.5):").grid(row=0, column=0, sticky="w")
        self.dp_entry = ttk.Entry(params_frame, width=8)
        self.dp_entry.insert(0, str(self.dp))
        self.dp_entry.grid(row=0, column=1, padx=5)
        ttk.Label(params_frame, text="Точность поиска", foreground="gray", font=("Arial", 8)).grid(row=0, column=2,
                                                                                                   columnspan=2,
                                                                                                   sticky="w", padx=5)

        # Строка 2
        ttk.Label(params_frame, text="minDist (пикс):").grid(row=1, column=0, sticky="w")
        self.minDist_entry = ttk.Entry(params_frame, width=8)
        self.minDist_entry.insert(0, str(self.minDist))
        self.minDist_entry.grid(row=1, column=1, padx=5)
        ttk.Label(params_frame, text="Мин.расстояние между центрами", foreground="gray", font=("Arial", 8)).grid(row=1,
                                                                                                                 column=2,
                                                                                                                 columnspan=2,
                                                                                                                 sticky="w",
                                                                                                                 padx=5)

        # Строка 3
        ttk.Label(params_frame, text="param1:").grid(row=2, column=0, sticky="w")
        self.param1_entry = ttk.Entry(params_frame, width=8)
        self.param1_entry.insert(0, str(self.param1))
        self.param1_entry.grid(row=2, column=1, padx=5)
        ttk.Label(params_frame, text="Чувствительность границ", foreground="gray", font=("Arial", 8)).grid(row=2,
                                                                                                           column=2,
                                                                                                           columnspan=2,
                                                                                                           sticky="w",
                                                                                                           padx=5)

        # Строка 4
        ttk.Label(params_frame, text="param2 (главный):").grid(row=3, column=0, sticky="w")
        self.param2_entry = ttk.Entry(params_frame, width=8)
        self.param2_entry.insert(0, str(self.param2))
        self.param2_entry.grid(row=3, column=1, padx=5)
        ttk.Label(params_frame, text="Чем МЕНЬШЕ, тем БОЛЬШЕ кругов", foreground="gray", font=("Arial", 8)).grid(row=3,
                                                                                                                 column=2,
                                                                                                                 columnspan=2,
                                                                                                                 sticky="w",
                                                                                                                 padx=5)

        # Строка 5
        ttk.Label(params_frame, text="minRadius (пикс):").grid(row=4, column=0, sticky="w")
        self.minRad_entry = ttk.Entry(params_frame, width=8)
        self.minRad_entry.insert(0, str(self.minRadius))
        self.minRad_entry.grid(row=4, column=1, padx=5)
        ttk.Label(params_frame, text="Мин.радиус отверстия", foreground="gray", font=("Arial", 8)).grid(row=4, column=2,
                                                                                                        columnspan=2,
                                                                                                        sticky="w",
                                                                                                        padx=5)

        # Строка 6
        ttk.Label(params_frame, text="maxRadius (пикс):").grid(row=5, column=0, sticky="w")
        self.maxRad_entry = ttk.Entry(params_frame, width=8)
        self.maxRad_entry.insert(0, str(self.maxRadius))
        self.maxRad_entry.grid(row=5, column=1, padx=5)
        ttk.Label(params_frame, text="Макс.радиус отверстия", foreground="gray", font=("Arial", 8)).grid(row=5,
                                                                                                         column=2,
                                                                                                         columnspan=2,
                                                                                                         sticky="w",
                                                                                                         padx=5)

        # Кнопка применения
        ttk.Button(params_frame, text="Применить настройки", command=self.update_params).grid(row=6, column=0,
                                                                                                columnspan=4, pady=8)

        # Блок автонастройки
        auto_frame = ttk.LabelFrame(right_frame, text="Автонастройка параметров", padding=5)
        auto_frame.pack(fill='x', pady=10)

        # Чекбокс включения автонастройки
        self.auto_tune_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(auto_frame, text="Включить автонастройку",
                        variable=self.auto_tune_var,
                        command=self.toggle_auto_tune).grid(row=0, column=0, columnspan=3, sticky="w", pady=5)

        # Целевое количество отверстий
        ttk.Label(auto_frame, text="Целевое кол-во отверстий:").grid(row=1, column=0, sticky="w")
        self.target_entry = ttk.Entry(auto_frame, width=10)
        self.target_entry.insert(0, str(self.target_holes))
        self.target_entry.grid(row=1, column=1, padx=5)
        ttk.Button(auto_frame, text="Применить", command=self.update_target_holes).grid(row=1, column=2, padx=5)

        # Статус автонастройки
        self.auto_status_label = ttk.Label(auto_frame, text="", foreground="blue")
        self.auto_status_label.grid(row=2, column=0, columnspan=3, sticky="w", pady=5)

    def toggle_auto_tune(self):
        """Включение/выключение автонастройки"""
        self.auto_tune_enabled = self.auto_tune_var.get()
        if self.auto_tune_enabled:
            self.auto_status_label.config(text="Автонастройка ВКЛЮЧЕНА", foreground="green")
        else:
            self.auto_status_label.config(text="Автонастройка ВЫКЛЮЧЕНА", foreground="gray")

    def update_target_holes(self):
        """Обновление целевого количества отверстий"""
        try:
            self.target_holes = int(self.target_entry.get())
            self.auto_status_label.config(text=f"Целевое кол-во: {self.target_holes}", foreground="blue")
        except ValueError:
            messagebox.showerror("Ошибка", "Введите корректное число")

    def auto_tune_params(self, image):
        """Двухэтапный автоматический подбор параметров"""
        if image is None or not self.auto_tune_enabled:
            return None, []

        self.is_tuning = True
        self.status_label.config(text="Статус: автонастройка...", foreground="orange")
        self.auto_status_label.config(text="Этап 1: грубый поиск...", foreground="orange")
        self.root.update()

        best_params = None
        best_count = 0
        best_result = None
        best_coords = []
        min_diff = float('inf')

        # ============ ЭТАП 1: ГРУБЫЙ ПОИСК ============
        # Диапазоны с большим шагом
        dp_range = [0.7, 0.8, 0.9]
        minDist_range = list(range(10, 61, 10))  # 10, 20, 30, 40, 50, 60
        param1_range = list(range(20, 51, 10))  # 20, 30, 40, 50
        param2_range = list(range(15, 48, 5))  # 15, 20, 25, 30, 35, 40, 45
        minRadius_range = list(range(5, 31, 5))  # 5, 10, 15, 20, 25, 30
        maxRadius_range = list(range(15, 56, 10))  # 15, 25, 35, 45, 55

        total_combinations = len(dp_range) * len(minDist_range) * len(param1_range) * len(param2_range) * len(
            minRadius_range) * len(maxRadius_range)
        combo_count = 0

        for dp in dp_range:
            for minDist in minDist_range:
                for param1 in param1_range:
                    for param2 in param2_range:
                        for minR in minRadius_range:
                            for maxR in maxRadius_range:
                                if minR >= maxR:
                                    continue

                                combo_count += 1
                                if combo_count % 20 == 0:
                                    self.auto_status_label.config(
                                        text=f"Грубый поиск: {combo_count}/{total_combinations}...",
                                        foreground="orange")
                                    self.root.update()

                                self.dp = dp
                                self.minDist = minDist
                                self.param1 = param1
                                self.param2 = param2
                                self.minRadius = minR
                                self.maxRadius = maxR

                                result_img, coords = self.find_holes_no_status(image)
                                count = len(coords)
                                diff = abs(count - self.target_holes)

                                if count == self.target_holes:
                                    best_params = (dp, minDist, param1, param2, minR, maxR)
                                    best_count = count
                                    best_result = result_img
                                    best_coords = coords
                                    min_diff = 0
                                    break

                                if diff < min_diff:
                                    min_diff = diff
                                    best_params = (dp, minDist, param1, param2, minR, maxR)
                                    best_count = count
                                    best_result = result_img
                                    best_coords = coords

                            if min_diff == 0:
                                break
                        if min_diff == 0:
                            break
                    if min_diff == 0:
                        break
                if min_diff == 0:
                    break
            if min_diff == 0:
                break

        # Если нашли точное количество, сразу возвращаем
        if min_diff == 0 and best_params:
            self.apply_best_params(best_params, best_count, best_result, best_coords)
            return best_result, best_coords

        # ============ ЭТАП 2: ТОЧНЫЙ ПОИСК ============
        if best_params:
            self.auto_status_label.config(text="Этап 2: точная настройка...", foreground="orange")
            self.root.update()

            # Извлекаем лучшие грубые параметры
            best_dp, best_minDist, best_param1, best_param2, best_minR, best_maxR = best_params

            # Точные диапазоны вокруг лучших грубых значений
            dp_range_fine = [best_dp - 0.1, best_dp, best_dp + 0.1]
            dp_range_fine = [x for x in dp_range_fine if 0.5 <= x <= 1.5]

            minDist_range_fine = list(range(max(2, best_minDist - 8), min(61, best_minDist + 9), 2))
            param1_range_fine = list(range(max(15, best_param1 - 10), min(55, best_param1 + 11), 5))
            param2_range_fine = list(range(max(15, best_param2 - 8), min(48, best_param2 + 9), 1))
            minRadius_range_fine = list(range(max(3, best_minR - 5), min(29, best_minR + 6), 2))
            maxRadius_range_fine = list(range(max(15, best_maxR - 10), min(56, best_maxR + 11), 5))

            fine_total = len(dp_range_fine) * len(minDist_range_fine) * len(param1_range_fine) * len(
                param2_range_fine) * len(minRadius_range_fine) * len(maxRadius_range_fine)
            fine_count = 0

            for dp in dp_range_fine:
                for minDist in minDist_range_fine:
                    for param1 in param1_range_fine:
                        for param2 in param2_range_fine:
                            for minR in minRadius_range_fine:
                                for maxR in maxRadius_range_fine:
                                    if minR >= maxR:
                                        continue

                                    fine_count += 1
                                    if fine_count % 20 == 0:
                                        self.auto_status_label.config(
                                            text=f"Точный поиск: {fine_count}/{fine_total}...",
                                            foreground="orange")
                                        self.root.update()

                                    self.dp = dp
                                    self.minDist = minDist
                                    self.param1 = param1
                                    self.param2 = param2
                                    self.minRadius = minR
                                    self.maxRadius = maxR

                                    result_img, coords = self.find_holes_no_status(image)
                                    count = len(coords)
                                    diff = abs(count - self.target_holes)

                                    if count == self.target_holes:
                                        best_params = (dp, minDist, param1, param2, minR, maxR)
                                        best_count = count
                                        best_result = result_img
                                        best_coords = coords
                                        min_diff = 0
                                        break

                                    if diff < min_diff:
                                        min_diff = diff
                                        best_params = (dp, minDist, param1, param2, minR, maxR)
                                        best_count = count
                                        best_result = result_img
                                        best_coords = coords

                                if min_diff == 0:
                                    break
                            if min_diff == 0:
                                break
                        if min_diff == 0:
                            break
                    if min_diff == 0:
                        break
                if min_diff == 0:
                    break

        # Применяем лучшие параметры
        if best_params:
            self.apply_best_params(best_params, best_count, best_result, best_coords)
            return best_result, best_coords

        self.is_tuning = False
        self.status_label.config(text="Статус: автонастройка не дала результата", foreground="red")
        self.auto_status_label.config(text="Не удалось подобрать параметры", foreground="red")
        return None, []

    def apply_best_params(self, params, count, result, coords):
        """Применяет лучшие найденные параметры"""
        self.dp, self.minDist, self.param1, self.param2, self.minRadius, self.maxRadius = params

        # Обновляем поля ввода
        self.dp_entry.delete(0, tk.END)
        self.dp_entry.insert(0, str(self.dp))
        self.minDist_entry.delete(0, tk.END)
        self.minDist_entry.insert(0, str(self.minDist))
        self.param1_entry.delete(0, tk.END)
        self.param1_entry.insert(0, str(self.param1))
        self.param2_entry.delete(0, tk.END)
        self.param2_entry.insert(0, str(self.param2))
        self.minRad_entry.delete(0, tk.END)
        self.minRad_entry.insert(0, str(self.minRadius))
        self.maxRad_entry.delete(0, tk.END)
        self.maxRad_entry.insert(0, str(self.maxRadius))

        self.auto_status_label.config(
            text=f"Найдено: {count} (цель: {self.target_holes}) | "
                 f"param2={self.param2}, minDist={self.minDist}",
            foreground="green")
        self.status_label.config(
            text=f"Статус: автонастройка завершена ({count} отверстий)",
            foreground="green")

        self.is_tuning = False

    def find_holes_no_status(self, image):
        """Поиск отверстий без обновления статуса (для автонастройки)"""
        if image is None:
            return None, []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.medianBlur(gray, 9)

        circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT,
                                   self.dp, self.minDist,
                                   param1=self.param1, param2=self.param2,
                                   minRadius=self.minRadius, maxRadius=self.maxRadius)

        result_image = image.copy()
        coordinates = []

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                center = (circle[0], circle[1])
                radius = circle[2]
                cv2.circle(result_image, center, radius, (0, 255, 0), 2)
                cv2.circle(result_image, center, 2, (0, 0, 255), 3)
                coordinates.append((center[0], center[1], radius))

        return result_image, coordinates

    def switch_mode(self):
        """Переключение между режимами фото и камеры"""
        mode = self.mode_var.get()

        # Останавливаем камеру если была запущена
        if self.camera:
            self.stop_camera()

        # Закрываем окно снимка если открыто
        if self.snapshot_window:
            self.snapshot_window.destroy()
            self.snapshot_window = None

        if mode == "photo":
            self.photo_frame.pack(fill='x', pady=5)
            self.camera_frame.pack_forget()
            self.status_label.config(text="Статус: режим фото", foreground="blue")
        else:
            self.camera_frame.pack(fill='x', pady=5)
            self.photo_frame.pack_forget()
            self.status_label.config(text="Статус: режим камеры", foreground="blue")

    def update_params(self):
        """Обновление параметров поиска"""
        try:
            self.dp = float(self.dp_entry.get())
            self.minDist = int(self.minDist_entry.get())
            self.param1 = int(self.param1_entry.get())
            self.param2 = int(self.param2_entry.get())
            self.minRadius = int(self.minRad_entry.get())
            self.maxRadius = int(self.maxRad_entry.get())
            self.status_label.config(text="Статус: параметры обновлены", foreground="green")
        except ValueError:
            messagebox.showerror("Ошибка", "Неверный формат параметров")

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            self.current_image = cv2.imread(file_path)
            self.display_image(self.current_image)
            self.status_label.config(text=f"Статус: загружено {os.path.basename(file_path)}", foreground="green")
            self.coords_text.delete(1.0, tk.END)

    def display_image(self, image):
        if image is None:
            return

        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image

        height, width = image_rgb.shape[:2]
        max_width = 750
        max_height = 650

        if width > max_width or height > max_height:
            scale = min(max_width / width, max_height / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image_rgb = cv2.resize(image_rgb, (new_width, new_height))

        img = Image.fromarray(image_rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        self.image_label.imgtk = imgtk
        self.image_label.config(image=imgtk)

    def find_holes(self, image):
        if image is None:
            return None, []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.medianBlur(gray, 9)

        circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT,
                                   self.dp, self.minDist,
                                   param1=self.param1, param2=self.param2,
                                   minRadius=self.minRadius, maxRadius=self.maxRadius)

        result_image = image.copy()
        coordinates = []

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                center = (circle[0], circle[1])
                radius = circle[2]
                cv2.circle(result_image, center, radius, (0, 255, 0), 2)
                cv2.circle(result_image, center, 2, (0, 0, 255), 3)
                coordinates.append((center[0], center[1], radius))
        else:
            self.status_label.config(text="Статус: отверстия не найдены", foreground="orange")

        return result_image, coordinates

    def process_image(self):
        if self.current_image is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите изображение")
            return

        # Если включена автонастройка, запускаем её
        if self.auto_tune_enabled:
            self.status_label.config(text="Статус: автонастройка...", foreground="orange")
            self.root.update()

            # Сначала делаем копию параметров для сохранения
            old_params = (self.dp, self.minDist, self.param1, self.param2, self.minRadius, self.maxRadius)

            result_image, coordinates = self.auto_tune_params(self.current_image)

            if result_image is not None:
                self.processed_image = result_image
                self.display_image(result_image)
                self.display_coordinates(coordinates)

                if coordinates:
                    self.status_label.config(
                        text=f"Статус: найдено {len(coordinates)} отверстий (автонастройка)",
                        foreground="green")
                else:
                    self.status_label.config(text="Статус: отверстия не найдены", foreground="orange")
            return

        # Обычный режим (без автонастройки)
        self.status_label.config(text="Статус: поиск отверстий...", foreground="orange")
        self.root.update()

        thread = threading.Thread(target=self._process_image_thread)
        thread.daemon = True
        thread.start()

    def _process_image_thread(self):
        result_image, coordinates = self.find_holes(self.current_image)

        if result_image is not None:
            self.processed_image = result_image
            self.root.after(0, lambda: self.display_image(result_image))
            self.root.after(0, lambda: self.display_coordinates(coordinates))

            if coordinates:
                self.root.after(0, lambda: self.status_label.config(
                    text=f"Статус: найдено {len(coordinates)} отверстий", foreground="green"))
            else:
                self.root.after(0, lambda: self.status_label.config(
                    text="Статус: отверстия не найдены", foreground="orange"))

    def display_coordinates(self, coordinates):
        self.coords_text.delete(1.0, tk.END)
        if coordinates:
            self.coords_text.insert(tk.END, f"Всего отверстий: {len(coordinates)}\n")
            self.coords_text.insert(tk.END, "-" * 35 + "\n")
            self.coords_text.insert(tk.END, "  N  |   X   |   Y   |  R\n")
            self.coords_text.insert(tk.END, "-" * 35 + "\n")
            for i, (x, y, r) in enumerate(coordinates, 1):
                self.coords_text.insert(tk.END, f"{i:3} | {x:4} | {y:4} | {r:2}\n")
        else:
            self.coords_text.insert(tk.END, "Отверстия не найдены")

    def toggle_camera(self):
        if self.camera is None:
            self.start_camera()
        else:
            self.stop_camera()

    def start_camera(self):
        self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not self.camera.isOpened():
            messagebox.showerror("Ошибка", "Камера не найдена!")
            self.camera = None
            return

        self.is_running = True
        self.cam_btn.config(text="Остановить камеру")
        self.status_label.config(text="Статус: камера работает", foreground="green")
        self.update_camera()

    def stop_camera(self):
        self.is_running = False
        if self.camera:
            self.camera.release()
            self.camera = None
        self.cam_btn.config(text="Запустить камеру")
        self.status_label.config(text="Статус: камера остановлена", foreground="blue")

    def update_camera(self):
        if not self.is_running or self.camera is None:
            return

        ret, frame = self.camera.read()
        if ret:
            self.current_image = frame
            self.display_image(frame)

        self.root.after(30, self.update_camera)

    def capture_and_process(self):
        if self.camera is None:
            messagebox.showwarning("Предупреждение", "Сначала запустите камеру")
            return

        if self.current_image is None:
            messagebox.showwarning("Предупреждение", "Нет изображения с камеры")
            return

        # Если включена автонастройка
        if self.auto_tune_enabled:
            self.status_label.config(text="Статус: автонастройка...", foreground="orange")
            self.root.update()

            # Останавливаем видео
            self.is_running = False

            # Выполняем автонастройку
            result_image, coordinates = self.auto_tune_params(self.current_image)

            if result_image is not None:
                self.processed_image = result_image
                self.display_image(result_image)
                self.display_coordinates(coordinates)

                if coordinates:
                    self.status_label.config(
                        text=f"Статус: найдено {len(coordinates)} отверстий (автонастройка)",
                        foreground="green")
                    self.show_snapshot_window(result_image, coordinates)
                else:
                    self.status_label.config(text="Статус: отверстия не найдены", foreground="orange")
                    self.is_running = True
                    self.update_camera()
            else:
                self.is_running = True
                self.update_camera()
            return

        # Обычный режим (без автонастройки)
        # Останавливаем обновление видео
        self.is_running = False

        self.status_label.config(text="Статус: обработка снимка...", foreground="orange")
        self.root.update()

        # Обрабатываем текущий кадр
        result_image, coordinates = self.find_holes(self.current_image)

        if result_image is not None:
            self.processed_image = result_image
            self.display_image(result_image)
            self.display_coordinates(coordinates)

            if coordinates:
                self.status_label.config(text=f"Статус: найдено {len(coordinates)} отверстий", foreground="green")
                self.show_snapshot_window(result_image, coordinates)
            else:
                self.status_label.config(text="Статус: отверстия не найдены", foreground="orange")
                self.is_running = True
                self.update_camera()

    def show_snapshot_window(self, image, coordinates):
        """Открывает отдельное окно с обработанным снимком"""
        if self.snapshot_window:
            self.snapshot_window.destroy()
            self.snapshot_window = None

        self.snapshot_window = tk.Toplevel(self.root)
        self.snapshot_window.title("Результат анализа снимка")
        self.snapshot_window.geometry("800x600")

        info_label = ttk.Label(self.snapshot_window,
                               text=f"Найдено отверстий: {len(coordinates)}",
                               font=("Arial", 12, "bold"))
        info_label.pack(pady=5)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]
        max_width = 750
        max_height = 500

        if width > max_width or height > max_height:
            scale = min(max_width / width, max_height / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image_rgb = cv2.resize(image_rgb, (new_width, new_height))

        img = Image.fromarray(image_rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        image_label = ttk.Label(self.snapshot_window, background="black")
        image_label.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        image_label.imgtk = imgtk
        image_label.config(image=imgtk)

        btn_frame = ttk.Frame(self.snapshot_window)
        btn_frame.pack(pady=10)

        ttk.Button(btn_frame, text="Закрыть и вернуться к видео",
                   command=self.close_snapshot).pack(side=tk.LEFT, padx=5)

        ttk.Button(btn_frame, text="Сохранить снимок",
                   command=lambda: self.save_snapshot(image)).pack(side=tk.LEFT, padx=5)

        self.snapshot_window.protocol("WM_DELETE_WINDOW", self.close_snapshot)

    def close_snapshot(self):
        """Закрывает окно снимка и возвращает видео"""
        if self.snapshot_window:
            self.snapshot_window.destroy()
            self.snapshot_window = None

        self.is_running = True
        if self.camera:
            self.status_label.config(text="Статус: камера работает", foreground="green")
            self.update_camera()

    def save_snapshot(self, image):
        """Сохраняет снимок с отверстиями"""
        if image is None:
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png")]
        )
        if file_path:
            cv2.imwrite(file_path, image)
            self.status_label.config(text="Статус: снимок сохранен", foreground="green")

    def save_result(self):
        if self.processed_image is None:
            messagebox.showwarning("Предупреждение", "Нет обработанного изображения для сохранения")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png")]
        )
        if file_path:
            cv2.imwrite(file_path, self.processed_image)
            self.status_label.config(text=f"Статус: сохранено в {os.path.basename(file_path)}", foreground="green")

    def on_closing(self):
        self.is_running = False
        if self.camera:
            self.camera.release()
        if self.snapshot_window:
            self.snapshot_window.destroy()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = HoleDetectionGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()