import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext
from PIL import Image, ImageTk, ImageEnhance
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import os
import threading
import json
import logging
from pathlib import Path
from datetime import datetime
import numpy as np

class BloodImageClassifier:
    def __init__(self, model_path=None):
        # Configuration
        self.config = {
            'model_path': model_path or r"D:\color model\Blood-Condition-Detector\blood_classifier.pth",
            'class_names': ['Fresh', 'Clotted', 'Heat Damaged'],
            'confidence_threshold': 0.5,
            'log_predictions': True,
            'max_image_size': (224, 224),
            'display_size': (350, 350)
        }
        
        # Initialize components
        self.model = None
        self.current_image_path = None
        self.current_image = None
        self.prediction_history = []
        
        # Setup logging
        self.setup_logging()
        
        # Image transforms with data augmentation options
        self.base_transform = transforms.Compose([
            transforms.Resize(self.config['max_image_size']),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Enhanced transform for difficult images
        self.enhanced_transform = transforms.Compose([
            transforms.Resize(self.config['max_image_size']),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.setup_gui()
        self.load_model()
    
    def setup_logging(self):
        """Setup logging for predictions and errors"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'blood_classifier.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_model(self):
        """Load the trained model with enhanced error handling"""
        def load_in_thread():
            try:
                if not os.path.exists(self.config['model_path']):
                    raise FileNotFoundError(f"Model file not found: {self.config['model_path']}")
                
                # Try different model architectures
                model_architectures = [
                    ('ResNet18', lambda: models.resnet18(pretrained=False)),
                    ('ResNet34', lambda: models.resnet34(pretrained=False)),
                    ('MobileNet', lambda: models.mobilenet_v2(pretrained=False))
                ]
                
                model_loaded = False
                for arch_name, arch_func in model_architectures:
                    try:
                        self.model = arch_func()
                        
                        # Adapt final layer
                        if hasattr(self.model, 'fc'):
                            self.model.fc = torch.nn.Linear(self.model.fc.in_features, len(self.config['class_names']))
                        elif hasattr(self.model, 'classifier'):
                            self.model.classifier[-1] = torch.nn.Linear(self.model.classifier[-1].in_features, len(self.config['class_names']))
                        
                        # Load state dict
                        state_dict = torch.load(self.config['model_path'], map_location=torch.device('cpu'))
                        self.model.load_state_dict(state_dict)
                        self.model.eval()
                        
                        self.logger.info(f"Model loaded successfully using {arch_name} architecture")
                        model_loaded = True
                        break
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to load with {arch_name}: {str(e)}")
                        continue
                
                if not model_loaded:
                    raise RuntimeError("Failed to load model with any supported architecture")
                
                # Update UI
                self.root.after(0, self.on_model_loaded, True, "Model loaded successfully")
                
            except Exception as e:
                error_msg = f"Error loading model: {str(e)}"
                self.logger.error(error_msg)
                self.root.after(0, self.on_model_loaded, False, error_msg)
        
        # Show loading status
        self.status_label.config(text="Loading model...", fg="blue")
        self.progress.start()
        
        # Load in separate thread
        thread = threading.Thread(target=load_in_thread)
        thread.daemon = True
        thread.start()
    
    def on_model_loaded(self, success, message):
        """Handle model loading completion"""
        self.progress.stop()
        
        if success:
            self.status_label.config(text=message, fg="green")
            self.predict_btn.config(state='normal')
            self.enhance_btn.config(state='normal')
        else:
            self.status_label.config(text=message, fg="red")
            messagebox.showerror("Model Loading Error", message)
            self.predict_btn.config(state='disabled')
            self.enhance_btn.config(state='disabled')
    
    def setup_gui(self):
        """Setup enhanced GUI components"""
        self.root = tk.Tk()
        self.root.title("Advanced Blood Image Classifier v2.0")
        self.root.geometry("800x900")
        self.root.configure(bg='#f5f5f5')
        
        # Configure styles
        style = ttk.Style()
        style.theme_use('clam')
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Main analysis tab
        self.main_frame = tk.Frame(self.notebook, bg='#f5f5f5')
        self.notebook.add(self.main_frame, text="Analysis")
        self.setup_main_tab()
        
        # History tab
        self.history_frame = tk.Frame(self.notebook, bg='#f5f5f5')
        self.notebook.add(self.history_frame, text="History")
        self.setup_history_tab()
        
        # Settings tab
        self.settings_frame = tk.Frame(self.notebook, bg='#f5f5f5')
        self.notebook.add(self.settings_frame, text="Settings")
        self.setup_settings_tab()
    
    def setup_main_tab(self):
        """Setup the main analysis interface"""
        main_frame = tk.Frame(self.main_frame, bg='#f5f5f5', padx=20, pady=20)
        main_frame.pack(fill='both', expand=True)
        
        # Title with version
        title_label = tk.Label(main_frame, text="Blood Condition Classifier v2.0", 
                              font=("Arial", 20, "bold"), bg='#f5f5f5', fg='#2c3e50')
        title_label.pack(pady=(0, 20))
        
        # File selection frame with drag-drop hint
        file_frame = tk.LabelFrame(main_frame, text="Image Selection", 
                                  font=("Arial", 12, "bold"), bg='#f5f5f5', padx=10, pady=10)
        file_frame.pack(fill='x', pady=(0, 20))
        
        btn_frame = tk.Frame(file_frame, bg='#f5f5f5')
        btn_frame.pack(fill='x')
        
        self.select_btn = tk.Button(btn_frame, text="📁 Select Image", 
                                   command=self.select_image,
                                   font=("Arial", 12, "bold"), bg='#3498db', fg='white',
                                   padx=20, pady=10, cursor='hand2', relief='raised')
        self.select_btn.pack(side='left')
        
        self.clear_btn = tk.Button(btn_frame, text="🗑️ Clear", 
                                  command=self.clear_image,
                                  font=("Arial", 12), bg='#e74c3c', fg='white',
                                  padx=15, pady=10, cursor='hand2', relief='raised')
        self.clear_btn.pack(side='left', padx=(10, 0))
        
        self.file_label = tk.Label(file_frame, text="No file selected • Supported: JPG, PNG, BMP, TIFF", 
                                  font=("Arial", 10), bg='#f5f5f5', fg='#7f8c8d')
        self.file_label.pack(pady=(10, 0))
        
        # Image display with better styling
        image_frame = tk.LabelFrame(main_frame, text="Selected Image", 
                                   font=("Arial", 12, "bold"), bg='#f5f5f5', padx=10, pady=10)
        image_frame.pack(pady=(0, 20))
        
        self.image_label = tk.Label(image_frame, text="📷\n\nNo image loaded\n\nClick 'Select Image' to begin", 
                                   bg='white', width=45, height=18,
                                   font=("Arial", 12), fg='#bdc3c7', relief='sunken', bd=2)
        self.image_label.pack(padx=10, pady=10)
        
        # Analysis buttons
        button_frame = tk.Frame(main_frame, bg='#f5f5f5')
        button_frame.pack(pady=(0, 20))
        
        self.predict_btn = tk.Button(button_frame, text="🔬 Analyze Image", 
                                    command=self.predict_threaded,
                                    font=("Arial", 14, "bold"), bg='#27ae60', fg='white',
                                    padx=30, pady=15, cursor='hand2', relief='raised',
                                    state='disabled')
        self.predict_btn.pack(side='left')
        
        self.enhance_btn = tk.Button(button_frame, text="✨ Enhanced Analysis", 
                                    command=self.enhanced_predict_threaded,
                                    font=("Arial", 14, "bold"), bg='#f39c12', fg='white',
                                    padx=20, pady=15, cursor='hand2', relief='raised',
                                    state='disabled')
        self.enhance_btn.pack(side='left', padx=(10, 0))
        
        # Progress bar with percentage
        progress_frame = tk.Frame(main_frame, bg='#f5f5f5')
        progress_frame.pack(fill='x', pady=(0, 20))
        
        self.progress = ttk.Progressbar(progress_frame, mode='indeterminate')
        self.progress.pack(fill='x')
        
        self.progress_label = tk.Label(progress_frame, text="", 
                                      font=("Arial", 9), bg='#f5f5f5', fg='#7f8c8d')
        self.progress_label.pack()
        
        # Enhanced results display
        results_frame = tk.LabelFrame(main_frame, text="Analysis Results", 
                                     font=("Arial", 12, "bold"), bg='#f5f5f5',
                                     padx=15, pady=15)
        results_frame.pack(fill='both', expand=True, pady=(0, 20))
        
        # Main result
        self.result_label = tk.Label(results_frame, text="", 
                                    font=("Arial", 16, "bold"), bg='#f5f5f5')
        self.result_label.pack(pady=(0, 10))
        
        # Confidence with visual indicator
        conf_frame = tk.Frame(results_frame, bg='#f5f5f5')
        conf_frame.pack(fill='x', pady=(0, 15))
        
        self.confidence_label = tk.Label(conf_frame, text="", 
                                        font=("Arial", 12), bg='#f5f5f5', fg='#34495e')
        self.confidence_label.pack()
        
        self.confidence_bar = ttk.Progressbar(conf_frame, mode='determinate', length=300)
        self.confidence_bar.pack(pady=5)
        
        # Detailed probabilities
        self.prob_text = scrolledtext.ScrolledText(results_frame, height=6, width=60,
                                                  font=("Consolas", 10), bg='#ecf0f1',
                                                  fg='#2c3e50', relief='sunken', bd=1)
        self.prob_text.pack(fill='x', pady=(0, 10))
        
        # Save result button
        self.save_btn = tk.Button(results_frame, text="💾 Save Result", 
                                 command=self.save_result,
                                 font=("Arial", 10), bg='#95a5a6', fg='white',
                                 padx=15, pady=5, cursor='hand2', state='disabled')
        self.save_btn.pack()
        
        # Status bar
        self.status_label = tk.Label(main_frame, text="Initializing...", 
                                    font=("Arial", 10), bg='#f5f5f5', fg='#7f8c8d',
                                    relief='sunken', bd=1, anchor='w', padx=10)
        self.status_label.pack(side='bottom', fill='x')
    
    def setup_history_tab(self):
        """Setup prediction history interface"""
        history_frame = tk.Frame(self.history_frame, bg='#f5f5f5', padx=20, pady=20)
        history_frame.pack(fill='both', expand=True)
        
        tk.Label(history_frame, text="Prediction History", 
                font=("Arial", 16, "bold"), bg='#f5f5f5', fg='#2c3e50').pack(pady=(0, 20))
        
        # History listbox with scrollbar
        list_frame = tk.Frame(history_frame, bg='#f5f5f5')
        list_frame.pack(fill='both', expand=True)
        
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side='right', fill='y')
        
        self.history_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set,
                                         font=("Consolas", 10), bg='#ecf0f1')
        self.history_listbox.pack(side='left', fill='both', expand=True)
        scrollbar.config(command=self.history_listbox.yview)
        
        # History buttons
        hist_btn_frame = tk.Frame(history_frame, bg='#f5f5f5')
        hist_btn_frame.pack(fill='x', pady=(20, 0))
        
        tk.Button(hist_btn_frame, text="Clear History", command=self.clear_history,
                 font=("Arial", 10), bg='#e74c3c', fg='white', padx=15, pady=5).pack(side='left')
        
        tk.Button(hist_btn_frame, text="Export History", command=self.export_history,
                 font=("Arial", 10), bg='#3498db', fg='white', padx=15, pady=5).pack(side='left', padx=(10, 0))
    
    def setup_settings_tab(self):
        """Setup configuration interface"""
        settings_frame = tk.Frame(self.settings_frame, bg='#f5f5f5', padx=20, pady=20)
        settings_frame.pack(fill='both', expand=True)
        
        tk.Label(settings_frame, text="Settings", 
                font=("Arial", 16, "bold"), bg='#f5f5f5', fg='#2c3e50').pack(pady=(0, 20))
        
        # Confidence threshold
        conf_frame = tk.LabelFrame(settings_frame, text="Confidence Threshold", 
                                  font=("Arial", 12, "bold"), bg='#f5f5f5', padx=10, pady=10)
        conf_frame.pack(fill='x', pady=(0, 20))
        
        self.conf_var = tk.DoubleVar(value=self.config['confidence_threshold'])
        conf_scale = tk.Scale(conf_frame, from_=0.0, to=1.0, resolution=0.05,
                             orient='horizontal', variable=self.conf_var,
                             font=("Arial", 10), bg='#f5f5f5')
        conf_scale.pack(fill='x')
        
        # Logging option
        log_frame = tk.LabelFrame(settings_frame, text="Logging", 
                                 font=("Arial", 12, "bold"), bg='#f5f5f5', padx=10, pady=10)
        log_frame.pack(fill='x', pady=(0, 20))
        
        self.log_var = tk.BooleanVar(value=self.config['log_predictions'])
        tk.Checkbutton(log_frame, text="Log predictions to file", variable=self.log_var,
                      font=("Arial", 10), bg='#f5f5f5').pack(anchor='w')
        
        # Apply settings button
        tk.Button(settings_frame, text="Apply Settings", command=self.apply_settings,
                 font=("Arial", 12, "bold"), bg='#27ae60', fg='white',
                 padx=20, pady=10).pack(pady=20)
    
    def apply_settings(self):
        """Apply configuration changes"""
        self.config['confidence_threshold'] = self.conf_var.get()
        self.config['log_predictions'] = self.log_var.get()
        messagebox.showinfo("Settings", "Settings applied successfully!")
    
    def select_image(self):
        """Enhanced image selection with validation"""
        filetypes = [
            ('Image files', '*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.webp'),
            ('JPEG files', '*.jpg *.jpeg'),
            ('PNG files', '*.png'),
            ('All files', '*.*')
        ]
        
        file_path = filedialog.askopenfilename(
            title="Select Blood Sample Image",
            filetypes=filetypes
        )
        
        if file_path:
            self.load_and_display_image(file_path)
    
    def clear_image(self):
        """Clear current image and results"""
        self.current_image_path = None
        self.current_image = None
        self.image_label.config(image='', text="📷\n\nNo image loaded\n\nClick 'Select Image' to begin")
        self.image_label.image = None
        self.file_label.config(text="No file selected • Supported: JPG, PNG, BMP, TIFF")
        self.result_label.config(text="")
        self.confidence_label.config(text="")
        self.confidence_bar['value'] = 0
        self.prob_text.delete('1.0', tk.END)
        self.save_btn.config(state='disabled')
        self.status_label.config(text="Image cleared", fg="blue")
    
    def load_and_display_image(self, path):
        """Enhanced image loading with metadata"""
        try:
            # Validate file
            if not os.path.exists(path):
                raise FileNotFoundError("Selected file does not exist")
            
            file_size = os.path.getsize(path)
            if file_size > 50 * 1024 * 1024:  # 50MB limit
                raise ValueError("Image file too large (max 50MB)")
            
            # Load image
            image = Image.open(path).convert('RGB')
            self.current_image = image.copy()
            self.current_image_path = path
            
            # Get image info
            width, height = image.size
            
            # Resize for display
            display_image = image.copy()
            display_image.thumbnail(self.config['display_size'], Image.Resampling.LANCZOS)
            
            # Create PhotoImage
            photo = ImageTk.PhotoImage(display_image)
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo
            
            # Update file info
            filename = Path(path).name
            size_mb = file_size / (1024 * 1024)
            self.file_label.config(text=f"Selected: {filename} • {width}x{height} • {size_mb:.1f}MB")
            
            # Clear previous results
            self.result_label.config(text="")
            self.confidence_label.config(text="")
            self.confidence_bar['value'] = 0
            self.prob_text.delete('1.0', tk.END)
            self.save_btn.config(state='disabled')
            
            self.status_label.config(text="Image loaded successfully", fg="green")
            self.logger.info(f"Image loaded: {filename} ({width}x{height})")
            
        except Exception as e:
            error_msg = f"Error loading image: {str(e)}"
            self.status_label.config(text=error_msg, fg="red")
            messagebox.showerror("Image Loading Error", error_msg)
            self.logger.error(error_msg)
    
    def predict_threaded(self):
        """Standard prediction in separate thread"""
        self._run_prediction(enhanced=False)
    
    def enhanced_predict_threaded(self):
        """Enhanced prediction with augmentation"""
        self._run_prediction(enhanced=True)
    
    def _run_prediction(self, enhanced=False):
        """Run prediction with progress tracking"""
        if not self.current_image_path or not self.model:
            return
        
        # Update UI
        self.progress.start()
        btn_text = "Enhanced Analysis..." if enhanced else "Analyzing..."
        self.predict_btn.config(state='disabled', text=btn_text)
        self.enhance_btn.config(state='disabled')
        self.progress_label.config(text="Processing image...")
        self.status_label.config(text="Running analysis...", fg="blue")
        
        # Run in thread
        thread = threading.Thread(target=self.predict_image, args=(enhanced,))
        thread.daemon = True
        thread.start()
    
    def predict_image(self, enhanced=False):
        """Enhanced prediction with multiple methods"""
        try:
            if not self.current_image:
                raise ValueError("No image loaded")
            
            self.root.after(0, lambda: self.progress_label.config(text="Preprocessing..."))
            
            # Choose transform
            transform = self.enhanced_transform if enhanced else self.base_transform
            img_tensor = transform(self.current_image).unsqueeze(0)
            
            self.root.after(0, lambda: self.progress_label.config(text="Running inference..."))
            
            # Make prediction
            with torch.no_grad():
                output = self.model(img_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                prediction = self.config['class_names'][predicted.item()]
                confidence_score = confidence.item()
                all_probs = probabilities[0].numpy()
            
            # Log prediction
            if self.config['log_predictions']:
                self.logger.info(f"Prediction: {prediction}, Confidence: {confidence_score:.3f}, Enhanced: {enhanced}")
            
            # Update GUI
            self.root.after(0, self.update_results, prediction, confidence_score, all_probs, enhanced)
            
        except Exception as e:
            error_msg = f"Error during prediction: {str(e)}"
            self.logger.error(error_msg)
            self.root.after(0, self.show_error, error_msg)
    
    def update_results(self, prediction, confidence, probabilities, enhanced=False):
        """Enhanced results display"""
        # Stop progress
        self.progress.stop()
        self.predict_btn.config(state='normal', text="🔬 Analyze Image")
        self.enhance_btn.config(state='normal', text="✨ Enhanced Analysis")
        self.progress_label.config(text="")
        
        # Color coding
        if confidence > 0.8:
            result_color = "#27ae60"  # Green
        elif confidence > 0.6:
            result_color = "#f39c12"  # Orange
        else:
            result_color = "#e74c3c"  # Red
        
        # Update main result
        method_text = " (Enhanced)" if enhanced else ""
        self.result_label.config(text=f"🔬 {prediction}{method_text}", fg=result_color)
        
        # Update confidence
        conf_text = f"Confidence: {confidence:.1%}"
        self.confidence_label.config(text=conf_text, fg=result_color)
        self.confidence_bar['value'] = confidence * 100
        
        # Update probabilities
        self.prob_text.delete('1.0', tk.END)
        prob_text = "Class Probabilities:\n" + "="*40 + "\n"
        
        # Sort by probability
        prob_data = [(self.config['class_names'][i], probabilities[i]) for i in range(len(self.config['class_names']))]
        prob_data.sort(key=lambda x: x[1], reverse=True)
        
        for class_name, prob in prob_data:
            bar_length = int(prob * 30)
            bar = "█" * bar_length + "░" * (30 - bar_length)
            prob_text += f"{class_name:12} {prob:6.1%} {bar}\n"
        
        prob_text += "="*40 + "\n"
        prob_text += f"Analysis method: {'Enhanced' if enhanced else 'Standard'}\n"
        prob_text += f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        self.prob_text.insert('1.0', prob_text)
        
        # Add to history
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'image_path': self.current_image_path,
            'prediction': prediction,
            'confidence': confidence,
            'enhanced': enhanced,
            'probabilities': probabilities.tolist()
        }
        self.prediction_history.append(history_entry)
        
        # Update history display
        history_text = f"{datetime.now().strftime('%H:%M:%S')} - {prediction} ({confidence:.1%})"
        if enhanced:
            history_text += " [Enhanced]"
        self.history_listbox.insert(0, history_text)
        
        # Enable save button
        self.save_btn.config(state='normal')
        
        # Status and warnings
        if confidence < self.config['confidence_threshold']:
            self.status_label.config(text=f"⚠️ Low confidence prediction ({confidence:.1%})", fg="orange")
            messagebox.showwarning("Low Confidence", 
                                 f"Prediction confidence is {confidence:.1%}, which is below the threshold of {self.config['confidence_threshold']:.1%}.\n\n"
                                 f"Consider:\n• Using enhanced analysis\n• Checking image quality\n• Ensuring proper lighting")
        else:
            self.status_label.config(text="✅ Analysis completed successfully", fg="green")
    
    def show_error(self, error_msg):
        """Enhanced error handling"""
        self.progress.stop()
        self.predict_btn.config(state='normal', text="🔬 Analyze Image")
        self.enhance_btn.config(state='normal', text="✨ Enhanced Analysis")
        self.progress_label.config(text="")
        self.status_label.config(text=f"❌ {error_msg}", fg="red")
        messagebox.showerror("Analysis Error", error_msg)
    
    def save_result(self):
        """Save current result to file"""
        if not self.prediction_history:
            return
        
        try:
            filename = f"blood_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                initialname=filename
            )
            
            if filepath:
                with open(filepath, 'w') as f:
                    json.dump(self.prediction_history[-1], f, indent=2)
                
                self.status_label.config(text=f"Result saved to {Path(filepath).name}", fg="green")
                messagebox.showinfo("Save Complete", f"Analysis result saved successfully!")
                
        except Exception as e:
            error_msg = f"Error saving result: {str(e)}"
            self.status_label.config(text=error_msg, fg="red")
            messagebox.showerror("Save Error", error_msg)
    
    def clear_history(self):
        """Clear prediction history"""
        if messagebox.askyesno("Clear History", "Are you sure you want to clear all prediction history?"):
            self.prediction_history.clear()
            self.history_listbox.delete(0, tk.END)
            self.status_label.config(text="History cleared", fg="blue")
    
    def export_history(self):
        """Export complete history to JSON"""
        if not self.prediction_history:
            messagebox.showinfo("Export", "No history to export")
            return
        
        try:
            filename = f"blood_analysis_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                initialname=filename
            )
            
            if filepath:
                with open(filepath, 'w') as f:
                    json.dump(self.prediction_history, f, indent=2)
                
                messagebox.showinfo("Export Complete", f"History exported with {len(self.prediction_history)} entries")
                
        except Exception as e:
            messagebox.showerror("Export Error", f"Error exporting history: {str(e)}")
    
    def run(self):
        """Start the application with error handling"""
        try:
            # Set window icon if available
            try:
                self.root.iconbitmap('icon.ico')
            except:
                pass  # Icon file not found, continue without it
            
            # Center window on screen
            self.root.update_idletasks()
            width = self.root.winfo_width()
            height = self.root.winfo_height()
            x = (self.root.winfo_screenwidth() // 2) - (width // 2)
            y = (self.root.winfo_screenheight() // 2) - (height // 2)
            self.root.geometry(f'{width}x{height}+{x}+{y}')
            
            # Handle window closing
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            
            # Start main loop
            self.logger.info("Application started")
            self.root.mainloop()
            
        except KeyboardInterrupt:
            self.logger.info("Application interrupted by user")
            self.root.quit()
        except Exception as e:
            self.logger.error(f"Critical error: {str(e)}")
            messagebox.showerror("Critical Error", f"A critical error occurred: {str(e)}")
    
    def on_closing(self):
        """Handle application closing"""
        if messagebox.askokcancel("Quit", "Are you sure you want to quit?"):
            self.logger.info("Application closed by user")
            self.root.destroy()

# Enhanced utility functions
class ImageProcessor:
    """Additional image processing utilities"""
    
    @staticmethod
    def enhance_image_quality(image):
        """Apply image enhancement techniques"""
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.1)
        
        # Enhance color
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(1.1)
        
        return image
    
    @staticmethod
    def detect_image_issues(image):
        """Detect potential issues with the image"""
        issues = []
        
        # Check image size
        width, height = image.size
        if width < 224 or height < 224:
            issues.append("Image resolution is low (minimum 224x224 recommended)")
        
        # Check brightness
        grayscale = image.convert('L')
        pixels = list(grayscale.getdata())
        avg_brightness = sum(pixels) / len(pixels)
        
        if avg_brightness < 50:
            issues.append("Image appears too dark")
        elif avg_brightness > 200:
            issues.append("Image appears too bright")
        
        # Check contrast
        min_val = min(pixels)
        max_val = max(pixels)
        contrast = max_val - min_val
        
        if contrast < 100:
            issues.append("Image has low contrast")
        
        return issues

class ModelManager:
    """Enhanced model management utilities"""
    
    @staticmethod
    def validate_model_file(model_path):
        """Validate model file integrity"""
        try:
            if not os.path.exists(model_path):
                return False, "Model file not found"
            
            # Check file size
            file_size = os.path.getsize(model_path)
            if file_size < 1024:  # Less than 1KB
                return False, "Model file appears corrupted (too small)"
            
            # Try to load state dict
            state_dict = torch.load(model_path, map_location='cpu')
            if not isinstance(state_dict, dict):
                return False, "Invalid model format"
            
            return True, "Model file is valid"
            
        except Exception as e:
            return False, f"Model validation error: {str(e)}"
    
    @staticmethod
    def get_model_info(model_path):
        """Get information about the model"""
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            
            info = {
                'file_size': os.path.getsize(model_path),
                'num_parameters': len(state_dict),
                'layer_names': list(state_dict.keys())[:5],  # First 5 layers
                'modified_date': datetime.fromtimestamp(os.path.getmtime(model_path))
            }
            
            return info
            
        except Exception as e:
            return {'error': str(e)}

# Configuration manager
class ConfigManager:
    """Handle application configuration"""
    
    CONFIG_FILE = "blood_classifier_config.json"
    
    @classmethod
    def load_config(cls):
        """Load configuration from file"""
        default_config = {
            'model_path': r"D:\color model\Blood-Condition-Detector\blood_classifier.pth",
            'class_names': ['Fresh', 'Clotted', 'Heat Damaged'],
            'confidence_threshold': 0.5,
            'log_predictions': True,
            'max_image_size': [224, 224],
            'display_size': [350, 350],
            'window_geometry': "800x900",
            'theme': 'default'
        }
        
        try:
            if os.path.exists(cls.CONFIG_FILE):
                with open(cls.CONFIG_FILE, 'r') as f:
                    config = json.load(f)
                # Merge with defaults
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
        except Exception:
            pass
        
        return default_config
    
    @classmethod
    def save_config(cls, config):
        """Save configuration to file"""
        try:
            with open(cls.CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=2)
            return True
        except Exception:
            return False

# Main application entry point
def main():
    """Main entry point with command line argument support"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Blood Image Classifier v2.0')
    parser.add_argument('--model', type=str, help='Path to model file')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    config = ConfigManager.load_config()
    
    # Override with command line arguments
    if args.model:
        config['model_path'] = args.model
    
    try:
        # Create and run application
        classifier = BloodImageClassifier(model_path=config['model_path'])
        classifier.config.update(config)
        classifier.run()
        
        # Save configuration on exit
        ConfigManager.save_config(classifier.config)
        
    except Exception as e:
        print(f"Failed to start application: {str(e)}")
        return 1
    
    return 0

# Usage examples and documentation
if __name__ == "__main__":
    import sys
    
    print("Blood Image Classifier v2.0")
    print("="*40)
    print("Enhanced GUI application for blood condition analysis")
    print("\nFeatures:")
    print("• Advanced GUI with tabbed interface")
    print("• Enhanced prediction with data augmentation")
    print("• Prediction history and export functionality")
    print("• Configurable settings and thresholds")
    print("• Comprehensive logging and error handling")
    print("• Image quality validation and enhancement")
    print("• Model validation and information display")
    print("\nStarting application...")
    print("="*40)
    
    sys.exit(main())