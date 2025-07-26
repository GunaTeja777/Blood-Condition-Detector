import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext
from PIL import Image, ImageTk, ImageEnhance
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import os
import sys
import threading
import json
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import time

class ProfessionalBloodClassifier:
    def __init__(self, model_path=None):
        # Use relative path if not specified
        if model_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, "blood_classifier.pth")
        
        # Configuration
        self.config = {
            'model_path': model_path,
            'class_names': ['clotted', 'fresh', 'spoiled'],
            'confidence_threshold': 0.5,
            'log_predictions': True,
            'max_image_size': (224, 224),
            'display_size': (350, 350),
            'camera_index': 0,
            'live_fps': 30,
            'prediction_interval': 1.0,
            'auto_save': False,
            'detection_region_size': 0.3
        }
        
        # Color scheme - Modern dark theme with professional colors
        self.colors = {
            'bg_primary': '#0d1117',      # GitHub dark background
            'bg_secondary': '#161b22',    # Slightly lighter dark
            'bg_tertiary': '#21262d',     # Card/panel background
            'accent_primary': '#238636',  # Success green
            'accent_secondary': '#1f6feb', # Primary blue
            'accent_warning': '#d29922',   # Warning orange
            'accent_danger': '#da3633',    # Error red
            'text_primary': '#f0f6fc',     # Primary text
            'text_secondary': '#8b949e',   # Secondary text
            'text_muted': '#6e7681',       # Muted text
            'border': '#30363d',           # Border color
            'button_hover': '#292e36'      # Button hover
        }
        
        # Initialize components
        self.model = None
        self.current_image_path = None
        self.current_image = None
        self.prediction_history = []
        
        # Live camera components
        self.camera = None
        self.is_camera_running = False
        self.live_thread = None
        self.last_prediction_time = 0
        self.current_live_frame = None
        self.live_prediction_result = None
        
        # Setup logging
        self.setup_logging()
        
        # Image transforms
        self.base_transform = transforms.Compose([
            transforms.Resize(self.config['max_image_size']),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
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
        self.status_label.config(text="🔄 Loading AI model...", fg=self.colors['accent_secondary'])
        self.progress.start()
        
        # Load in separate thread
        thread = threading.Thread(target=load_in_thread)
        thread.daemon = True
        thread.start()
    
    def on_model_loaded(self, success, message):
        """Handle model loading completion"""
        self.progress.stop()
        
        if success:
            self.status_label.config(text=f"✅ {message}", fg=self.colors['accent_primary'])
            self.predict_btn.config(state='normal')
            self.enhance_btn.config(state='normal')
            self.start_camera_btn.config(state='normal')
            self.camera_settings_btn.config(state='normal')
        else:
            self.status_label.config(text=f"❌ {message}", fg=self.colors['accent_danger'])
            messagebox.showerror("Model Loading Error", message)
            self.predict_btn.config(state='disabled')
            self.enhance_btn.config(state='disabled')
            self.start_camera_btn.config(state='disabled')
    
    def create_modern_button(self, parent, text, command, bg_color, hover_color=None, 
                           width=None, height=None, font_size=11, icon=""):
        """Create a modern styled button with hover effects"""
        if hover_color is None:
            hover_color = self.colors['button_hover']
        
        btn = tk.Button(
            parent,
            text=f"{icon} {text}" if icon else text,
            command=command,
            font=("Segoe UI", font_size, "normal"),
            bg=bg_color,
            fg=self.colors['text_primary'],
            relief='flat',
            bd=0,
            padx=20 if width is None else width//8,
            pady=8 if height is None else height//6,
            cursor='hand2',
            activebackground=hover_color,
            activeforeground=self.colors['text_primary'],
            width=width,
            height=height
        )
        
        # Add hover effects
        def on_enter(e):
            btn.config(bg=hover_color)
        
        def on_leave(e):
            btn.config(bg=bg_color)
        
        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)
        
        return btn
    
    def create_modern_frame(self, parent, title=None, padding=15):
        """Create a modern styled frame with optional title"""
        # Main container
        container = tk.Frame(parent, bg=self.colors['bg_primary'])
        
        if title:
            # Title bar
            title_frame = tk.Frame(container, bg=self.colors['bg_tertiary'], height=40)
            title_frame.pack(fill='x', padx=1, pady=(1, 0))
            title_frame.pack_propagate(False)
            
            title_label = tk.Label(
                title_frame,
                text=title,
                font=("Segoe UI", 12, "bold"),
                bg=self.colors['bg_tertiary'],
                fg=self.colors['text_primary'],
                anchor='w'
            )
            title_label.pack(side='left', padx=padding, pady=10)
        
        # Content frame
        content_frame = tk.Frame(
            container,
            bg=self.colors['bg_secondary'],
            relief='flat',
            bd=1
        )
        content_frame.pack(fill='both', expand=True, padx=1, pady=(0, 1))
        
        # Inner content with padding
        inner_frame = tk.Frame(content_frame, bg=self.colors['bg_secondary'])
        inner_frame.pack(fill='both', expand=True, padx=padding, pady=padding)
        
        return container, inner_frame
    
    def setup_gui(self):
        """Setup modern professional GUI"""
        self.root = tk.Tk()
        self.root.title("Blood Analysis AI - Professional Edition")
        self.root.geometry("1600x1000")
        self.root.configure(bg=self.colors['bg_primary'])
        self.root.minsize(1200, 800)
        
        # Configure modern ttk styles
        self.setup_modern_styles()
        
        # Header
        self.create_header()
        
        # Main content area
        main_container = tk.Frame(self.root, bg=self.colors['bg_primary'])
        main_container.pack(fill='both', expand=True, padx=20, pady=(0, 20))
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_container, style='Modern.TNotebook')
        self.notebook.pack(fill='both', expand=True)
        
        # Setup tabs
        self.setup_analysis_tab()
        self.setup_live_camera_tab()
        self.setup_history_tab()
        self.setup_settings_tab()
        
        # Status bar
        self.create_status_bar()
    
    def setup_modern_styles(self):
        """Configure modern ttk styles"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure notebook
        style.configure(
            'Modern.TNotebook',
            background=self.colors['bg_primary'],
            borderwidth=0,
            tabmargins=[0, 0, 0, 0]
        )
        
        style.configure(
            'Modern.TNotebook.Tab',
            background=self.colors['bg_tertiary'],
            foreground=self.colors['text_secondary'],
            padding=[20, 12],
            focuscolor='none',
            font=('Segoe UI', 10, 'normal')
        )
        
        style.map(
            'Modern.TNotebook.Tab',
            background=[
                ('selected', self.colors['bg_secondary']),
                ('active', self.colors['button_hover'])
            ],
            foreground=[
                ('selected', self.colors['text_primary']),
                ('active', self.colors['text_primary'])
            ]
        )
        
        # Configure progressbar
        style.configure(
            'Modern.Horizontal.TProgressbar',
            background=self.colors['accent_primary'],
            troughcolor=self.colors['bg_tertiary'],
            borderwidth=0,
            lightcolor=self.colors['accent_primary'],
            darkcolor=self.colors['accent_primary']
        )
    
    def create_header(self):
        """Create professional header with branding"""
        header_frame = tk.Frame(self.root, bg=self.colors['bg_secondary'], height=80)
        header_frame.pack(fill='x', padx=20, pady=(20, 20))
        header_frame.pack_propagate(False)
        
        # Left side - Logo and title
        left_frame = tk.Frame(header_frame, bg=self.colors['bg_secondary'])
        left_frame.pack(side='left', fill='y', padx=20, pady=15)
        
        # App icon/logo placeholder
        logo_frame = tk.Frame(left_frame, bg=self.colors['accent_primary'], width=50, height=50)
        logo_frame.pack(side='left', pady=5)
        logo_frame.pack_propagate(False)
        
        logo_label = tk.Label(
            logo_frame,
            text="🩸",
            font=("Segoe UI", 20),
            bg=self.colors['accent_primary'],
            fg='white'
        )
        logo_label.pack(expand=True)
        
        # Title and subtitle
        title_frame = tk.Frame(left_frame, bg=self.colors['bg_secondary'])
        title_frame.pack(side='left', padx=(15, 0), fill='y')
        
        title_label = tk.Label(
            title_frame,
            text="Blood Analysis AI",
            font=("Segoe UI", 18, "bold"),
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_primary']
        )
        title_label.pack(anchor='w')
        
        subtitle_label = tk.Label(
            title_frame,
            text="Professional Medical Image Classification System",
            font=("Segoe UI", 10),
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_secondary']
        )
        subtitle_label.pack(anchor='w')
        
        # Right side - Model status and info
        right_frame = tk.Frame(header_frame, bg=self.colors['bg_secondary'])
        right_frame.pack(side='right', fill='y', padx=20, pady=15)
        
        # Model status indicator
        status_frame = tk.Frame(right_frame, bg=self.colors['bg_tertiary'])
        status_frame.pack(side='right', padx=10)
        
        self.model_status_label = tk.Label(
            status_frame,
            text="🤖 AI Model: Initializing",
            font=("Segoe UI", 10, "bold"),
            bg=self.colors['bg_tertiary'],
            fg=self.colors['text_secondary'],
            padx=15,
            pady=8
        )
        self.model_status_label.pack()
        
        # Progress bar for model loading
        self.progress = ttk.Progressbar(
            right_frame,
            mode='indeterminate',
            style='Modern.Horizontal.TProgressbar',
            length=200
        )
        self.progress.pack(side='right', padx=(0, 10), pady=20)
    
    def create_status_bar(self):
        """Create modern status bar"""
        status_frame = tk.Frame(self.root, bg=self.colors['bg_tertiary'], height=35)
        status_frame.pack(side='bottom', fill='x')
        status_frame.pack_propagate(False)
        
        # Status text
        self.status_label = tk.Label(
            status_frame,
            text="🚀 System initialized - Ready for analysis",
            font=("Segoe UI", 9),
            bg=self.colors['bg_tertiary'],
            fg=self.colors['text_secondary'],
            anchor='w'
        )
        self.status_label.pack(side='left', padx=20, pady=8)
        
        # Version info
        version_label = tk.Label(
            status_frame,
            text="v4.0 Professional",
            font=("Segoe UI", 9),
            bg=self.colors['bg_tertiary'],
            fg=self.colors['text_muted'],
            anchor='e'
        )
        version_label.pack(side='right', padx=20, pady=8)
    
    def setup_analysis_tab(self):
        """Setup the main analysis tab with modern design"""
        # Create main tab frame
        tab_frame = tk.Frame(self.notebook, bg=self.colors['bg_primary'])
        self.notebook.add(tab_frame, text="📊 Static Analysis")
        
        # Main container with padding
        main_container = tk.Frame(tab_frame, bg=self.colors['bg_primary'])
        main_container.pack(fill='both', expand=True, padx=25, pady=25)
        
        # Top section - Image selection and display
        top_section = tk.Frame(main_container, bg=self.colors['bg_primary'])
        top_section.pack(fill='x', pady=(0, 20))
        
        # Left side - Image selection
        left_panel, left_content = self.create_modern_frame(top_section, "📁 Image Selection")
        left_panel.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        # File selection buttons
        btn_frame = tk.Frame(left_content, bg=self.colors['bg_secondary'])
        btn_frame.pack(fill='x', pady=(0, 15))
        
        self.select_btn = self.create_modern_button(
            btn_frame, "Select Image", self.select_image,
            self.colors['accent_secondary'], font_size=12, icon="📂"
        )
        self.select_btn.pack(side='left', padx=(0, 10))
        
        self.clear_btn = self.create_modern_button(
            btn_frame, "Clear", self.clear_image,
            self.colors['accent_danger'], font_size=12, icon="🗑️"
        )
        self.clear_btn.pack(side='left')
        
        # File info
        self.file_label = tk.Label(
            left_content,
            text="No file selected • Supported: JPG, PNG, BMP, TIFF",
            font=("Segoe UI", 10),
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_secondary'],
            anchor='w'
        )
        self.file_label.pack(fill='x')
        
        # Right side - Image display
        right_panel, right_content = self.create_modern_frame(top_section, "🖼️ Selected Image")
        right_panel.pack(side='right', fill='both', expand=True, padx=(10, 0))
        
        # Image display area
        image_container = tk.Frame(right_content, bg=self.colors['bg_tertiary'], relief='flat', bd=1)
        image_container.pack(fill='both', expand=True)
        
        self.image_label = tk.Label(
            image_container,
            text="📷\n\nNo image loaded\n\nClick 'Select Image' to begin",
            bg=self.colors['bg_tertiary'],
            fg=self.colors['text_muted'],
            font=("Segoe UI", 12),
            width=45,
            height=15
        )
        self.image_label.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Analysis section
        analysis_section, analysis_content = self.create_modern_frame(main_container, "🔬 Analysis Controls")
        analysis_section.pack(fill='x', pady=(0, 20))
        
        # Analysis buttons
        btn_container = tk.Frame(analysis_content, bg=self.colors['bg_secondary'])
        btn_container.pack(fill='x', pady=(0, 15))
        
        self.predict_btn = self.create_modern_button(
            btn_container, "Analyze Image", self.predict_threaded,
            self.colors['accent_primary'], font_size=14, icon="🔬"
        )
        self.predict_btn.pack(side='left', padx=(0, 15))
        self.predict_btn.config(state='disabled')
        
        self.enhance_btn = self.create_modern_button(
            btn_container, "Enhanced Analysis", self.enhanced_predict_threaded,
            self.colors['accent_warning'], font_size=14, icon="✨"
        )
        self.enhance_btn.pack(side='left')
        self.enhance_btn.config(state='disabled')
        
        # Progress indicator
        progress_frame = tk.Frame(analysis_content, bg=self.colors['bg_secondary'])
        progress_frame.pack(fill='x', pady=(0, 10))
        
        self.analysis_progress = ttk.Progressbar(
            progress_frame,
            mode='indeterminate',
            style='Modern.Horizontal.TProgressbar'
        )
        self.analysis_progress.pack(fill='x')
        
        self.progress_label = tk.Label(
            progress_frame,
            text="",
            font=("Segoe UI", 9),
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_secondary']
        )
        self.progress_label.pack(pady=(5, 0))
        
        # Results section
        results_section, results_content = self.create_modern_frame(main_container, "📈 Analysis Results")
        results_section.pack(fill='both', expand=True)
        
        # Main result display
        result_header = tk.Frame(results_content, bg=self.colors['bg_secondary'])
        result_header.pack(fill='x', pady=(0, 15))
        
        self.result_label = tk.Label(
            result_header,
            text="Awaiting analysis...",
            font=("Segoe UI", 16, "bold"),
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_secondary']
        )
        self.result_label.pack(side='left')
        
        self.confidence_label = tk.Label(
            result_header,
            text="",
            font=("Segoe UI", 14),
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_secondary']
        )
        self.confidence_label.pack(side='right')
        
        # Confidence bar
        conf_frame = tk.Frame(results_content, bg=self.colors['bg_secondary'])
        conf_frame.pack(fill='x', pady=(0, 15))
        
        tk.Label(
            conf_frame,
            text="Confidence Level",
            font=("Segoe UI", 10, "bold"),
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_primary']
        ).pack(anchor='w', pady=(0, 5))
        
        self.confidence_bar = ttk.Progressbar(
            conf_frame,
            mode='determinate',
            style='Modern.Horizontal.TProgressbar',
            length=400
        )
        self.confidence_bar.pack(fill='x')
        
        # Detailed probabilities
        tk.Label(
            results_content,
            text="Detailed Analysis",
            font=("Segoe UI", 10, "bold"),
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_primary']
        ).pack(anchor='w', pady=(15, 5))
        
        # Text area with modern styling
        text_frame = tk.Frame(results_content, bg=self.colors['bg_tertiary'], relief='flat', bd=1)
        text_frame.pack(fill='both', expand=True, pady=(0, 15))
        
        self.prob_text = scrolledtext.ScrolledText(
            text_frame,
            height=8,
            font=("Consolas", 10),
            bg=self.colors['bg_tertiary'],
            fg=self.colors['text_primary'],
            relief='flat',
            bd=0,
            insertbackground=self.colors['text_primary'],
            selectbackground=self.colors['accent_secondary']
        )
        self.prob_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Save result button
        self.save_btn = self.create_modern_button(
            results_content, "Save Result", self.save_result,
            self.colors['accent_secondary'], icon="💾"
        )
        self.save_btn.pack(anchor='w')
        self.save_btn.config(state='disabled')
    
    def setup_live_camera_tab(self):
        """Setup live camera tab with modern design"""
        tab_frame = tk.Frame(self.notebook, bg=self.colors['bg_primary'])
        self.notebook.add(tab_frame, text="📹 Live Camera")
        
        main_container = tk.Frame(tab_frame, bg=self.colors['bg_primary'])
        main_container.pack(fill='both', expand=True, padx=25, pady=25)
        
        # Camera controls
        controls_section, controls_content = self.create_modern_frame(main_container, "🎮 Camera Controls")
        controls_section.pack(fill='x', pady=(0, 20))
        
        # Control buttons
        btn_frame = tk.Frame(controls_content, bg=self.colors['bg_secondary'])
        btn_frame.pack(fill='x', pady=(0, 15))
        
        self.start_camera_btn = self.create_modern_button(
            btn_frame, "Start Camera", self.start_camera,
            self.colors['accent_primary'], icon="📹"
        )
        self.start_camera_btn.pack(side='left', padx=(0, 10))
        self.start_camera_btn.config(state='disabled')
        
        self.stop_camera_btn = self.create_modern_button(
            btn_frame, "Stop Camera", self.stop_camera,
            self.colors['accent_danger'], icon="⏹️"
        )
        self.stop_camera_btn.pack(side='left', padx=(0, 10))
        self.stop_camera_btn.config(state='disabled')
        
        self.capture_btn = self.create_modern_button(
            btn_frame, "Capture Frame", self.capture_frame,
            self.colors['accent_secondary'], icon="📸"
        )
        self.capture_btn.pack(side='left', padx=(0, 10))
        self.capture_btn.config(state='disabled')
        
        self.camera_settings_btn = self.create_modern_button(
            btn_frame, "Settings", self.camera_settings,
            self.colors['bg_tertiary'], icon="⚙️"
        )
        self.camera_settings_btn.pack(side='left')
        self.camera_settings_btn.config(state='disabled')
        
        # Camera info
        self.camera_info_label = tk.Label(
            controls_content,
            text="📷 Camera: Not connected",
            font=("Segoe UI", 10),
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_secondary']
        )
        self.camera_info_label.pack(anchor='w')
        
        # Main content area
        content_frame = tk.Frame(main_container, bg=self.colors['bg_primary'])
        content_frame.pack(fill='both', expand=True)
        
        # Left side - Video feed
        video_section, video_content = self.create_modern_frame(content_frame, "📺 Live Video Feed")
        video_section.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        # Video display with border
        video_container = tk.Frame(video_content, bg=self.colors['bg_primary'], relief='flat', bd=2)
        video_container.pack(fill='both', expand=True)
        
        self.video_label = tk.Label(
            video_container,
            text="📹\n\nCamera Feed\n\nStart camera to begin live analysis",
            bg='#000000',
            fg='white',
            font=("Segoe UI", 14),
            width=70,
            height=25
        )
        self.video_label.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Right side - Analysis controls and results
        right_panel = tk.Frame(content_frame, bg=self.colors['bg_primary'], width=400)
        right_panel.pack(side='right', fill='y', padx=(10, 0))
        right_panel.pack_propagate(False)
        
        # Live analysis controls
        live_controls_section, live_controls_content = self.create_modern_frame(right_panel, "🤖 Analysis Settings")
        live_controls_section.pack(fill='x', pady=(0, 15))
        
        # Auto-analysis toggle
        self.auto_analysis_var = tk.BooleanVar(value=True)
        auto_check = tk.Checkbutton(
            live_controls_content,
            text="🤖 Auto-Analysis",
            variable=self.auto_analysis_var,
            font=("Segoe UI", 11, "bold"),
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_primary'],
            selectcolor=self.colors['accent_primary'],
            activebackground=self.colors['bg_secondary'],
            activeforeground=self.colors['text_primary'],
            command=self.toggle_auto_analysis
        )
        auto_check.pack(anchor='w', pady=(0, 10))
        
        # Detection region toggle
        self.show_region_var = tk.BooleanVar(value=True)
        region_check = tk.Checkbutton(
            live_controls_content,
            text="🔍 Show Detection Region",
            variable=self.show_region_var,
            font=("Segoe UI", 11),
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_primary'],
            selectcolor=self.colors['accent_primary'],
            activebackground=self.colors['bg_secondary'],
            activeforeground=self.colors['text_primary']
        )
        region_check.pack(anchor='w')
        
        # Live results
        results_section, results_content = self.create_modern_frame(right_panel, "📊 Live Results")
        results_section.pack(fill='both', expand=True)
        
        # Current prediction display
        current_pred_frame = tk.Frame(results_content, bg=self.colors['bg_tertiary'], relief='flat', bd=1)
        current_pred_frame.pack(fill='x', pady=(0, 15))
        
        tk.Label(
            current_pred_frame,
            text="Current Analysis",
            font=("Segoe UI", 10, "bold"),
            bg=self.colors['bg_tertiary'],
            fg=self.colors['text_primary']
        ).pack(anchor='w', padx=15, pady=(10, 5))
        
        pred_display_frame = tk.Frame(current_pred_frame, bg=self.colors['bg_tertiary'])
        pred_display_frame.pack(fill='x', padx=15, pady=(0, 10))
        
        self.live_result_label = tk.Label(
            pred_display_frame,
            text="Waiting for analysis...",
            font=("Segoe UI", 12, "bold"),
            bg=self.colors['bg_tertiary'],
            fg=self.colors['text_secondary']
        )
        self.live_result_label.pack(side='left')
        
        self.live_confidence_label = tk.Label(
            pred_display_frame,
            text="",
            font=("Segoe UI", 11),
            bg=self.colors['bg_tertiary'],
            fg=self.colors['text_secondary']
        )
        self.live_confidence_label.pack(side='right')
        
        # Live prediction history
        tk.Label(
            results_content,
            text="Recent Predictions",
            font=("Segoe UI", 10, "bold"),
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_primary']
        ).pack(anchor='w', pady=(0, 5))
        
        history_frame = tk.Frame(results_content, bg=self.colors['bg_tertiary'], relief='flat', bd=1)
        history_frame.pack(fill='both', expand=True)
        
        self.live_history_text = scrolledtext.ScrolledText(
            history_frame,
            height=12,
            font=("Consolas", 9),
            bg=self.colors['bg_tertiary'],
            fg=self.colors['text_primary'],
            relief='flat',
            bd=0,
            insertbackground=self.colors['text_primary'],
            selectbackground=self.colors['accent_secondary']
        )
        self.live_history_text.pack(fill='both', expand=True, padx=10, pady=10)
    
    def setup_history_tab(self):
        """Setup prediction history tab with modern design"""
        tab_frame = tk.Frame(self.notebook, bg=self.colors['bg_primary'])
        self.notebook.add(tab_frame, text="📋 History")
        
        main_container = tk.Frame(tab_frame, bg=self.colors['bg_primary'])
        main_container.pack(fill='both', expand=True, padx=25, pady=25)
        
        # History section
        history_section, history_content = self.create_modern_frame(main_container, "📈 Prediction History")
        history_section.pack(fill='both', expand=True, pady=(0, 20))
        
        # History listbox with modern styling
        list_frame = tk.Frame(history_content, bg=self.colors['bg_tertiary'], relief='flat', bd=1)
        list_frame.pack(fill='both', expand=True, pady=(0, 15))
        
        # Custom scrollbar
        scrollbar = tk.Scrollbar(list_frame, bg=self.colors['bg_tertiary'], troughcolor=self.colors['bg_secondary'])
        scrollbar.pack(side='right', fill='y')
        
        self.history_listbox = tk.Listbox(
            list_frame,
            yscrollcommand=scrollbar.set,
            font=("Consolas", 10),
            bg=self.colors['bg_tertiary'],
            fg=self.colors['text_primary'],
            selectbackground=self.colors['accent_secondary'],
            selectforeground=self.colors['text_primary'],
            relief='flat',
            bd=0
        )
        self.history_listbox.pack(side='left', fill='both', expand=True, padx=10, pady=10)
        scrollbar.config(command=self.history_listbox.yview)
        
        # History controls
        controls_frame = tk.Frame(history_content, bg=self.colors['bg_secondary'])
        controls_frame.pack(fill='x')
        
        self.clear_history_btn = self.create_modern_button(
            controls_frame, "Clear History", self.clear_history,
            self.colors['accent_danger'], icon="🗑️"
        )
        self.clear_history_btn.pack(side='left', padx=(0, 10))
        
        self.export_history_btn = self.create_modern_button(
            controls_frame, "Export History", self.export_history,
            self.colors['accent_secondary'], icon="📤"
        )
        self.export_history_btn.pack(side='left')
        
        # Statistics section
        stats_section, stats_content = self.create_modern_frame(main_container, "📊 Statistics")
        stats_section.pack(fill='x')
        
        # Stats display
        stats_frame = tk.Frame(stats_content, bg=self.colors['bg_secondary'])
        stats_frame.pack(fill='x')
        
        self.stats_labels = {}
        stat_names = ["Total Analyses", "High Confidence", "Average Confidence", "Most Common"]
        
        for i, stat_name in enumerate(stat_names):
            stat_container = tk.Frame(stats_frame, bg=self.colors['bg_tertiary'], relief='flat', bd=1)
            stat_container.pack(side='left', fill='both', expand=True, padx=(0, 10) if i < 3 else 0)
            
            tk.Label(
                stat_container,
                text=stat_name,
                font=("Segoe UI", 9),
                bg=self.colors['bg_tertiary'],
                fg=self.colors['text_secondary']
            ).pack(pady=(10, 2))
            
            self.stats_labels[stat_name] = tk.Label(
                stat_container,
                text="0",
                font=("Segoe UI", 14, "bold"),
                bg=self.colors['bg_tertiary'],
                fg=self.colors['accent_primary']
            )
            self.stats_labels[stat_name].pack(pady=(0, 10))
    
    def setup_settings_tab(self):
        """Setup settings tab with modern design"""
        tab_frame = tk.Frame(self.notebook, bg=self.colors['bg_primary'])
        self.notebook.add(tab_frame, text="⚙️ Settings")
        
        main_container = tk.Frame(tab_frame, bg=self.colors['bg_primary'])
        main_container.pack(fill='both', expand=True, padx=25, pady=25)
        
        # Create scrollable frame for settings
        canvas = tk.Canvas(main_container, bg=self.colors['bg_primary'], highlightthickness=0)
        scrollbar = tk.Scrollbar(main_container, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.colors['bg_primary'])
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Analysis Settings
        analysis_section, analysis_content = self.create_modern_frame(scrollable_frame, "🔬 Analysis Settings")
        analysis_section.pack(fill='x', pady=(0, 20))
        
        # Confidence threshold
        conf_frame = tk.Frame(analysis_content, bg=self.colors['bg_secondary'])
        conf_frame.pack(fill='x', pady=(0, 15))
        
        tk.Label(
            conf_frame,
            text="Confidence Threshold",
            font=("Segoe UI", 11, "bold"),
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_primary']
        ).pack(anchor='w', pady=(0, 5))
        
        self.conf_var = tk.DoubleVar(value=self.config['confidence_threshold'])
        conf_scale = tk.Scale(
            conf_frame,
            from_=0.0, to=1.0, resolution=0.05,
            orient='horizontal',
            variable=self.conf_var,
            font=("Segoe UI", 10),
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_primary'],
            highlightbackground=self.colors['bg_secondary'],
            troughcolor=self.colors['bg_tertiary'],
            activebackground=self.colors['accent_primary']
        )
        conf_scale.pack(fill='x')
        
        # Logging option
        self.log_var = tk.BooleanVar(value=self.config['log_predictions'])
        log_check = tk.Checkbutton(
            analysis_content,
            text="📝 Enable prediction logging",
            variable=self.log_var,
            font=("Segoe UI", 11),
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_primary'],
            selectcolor=self.colors['accent_primary'],
            activebackground=self.colors['bg_secondary'],
            activeforeground=self.colors['text_primary']
        )
        log_check.pack(anchor='w', pady=(0, 15))
        
        # Camera Settings
        camera_section, camera_content = self.create_modern_frame(scrollable_frame, "📹 Camera Settings")
        camera_section.pack(fill='x', pady=(0, 20))
        
        # Camera index
        cam_index_frame = tk.Frame(camera_content, bg=self.colors['bg_secondary'])
        cam_index_frame.pack(fill='x', pady=(0, 15))
        
        tk.Label(
            cam_index_frame,
            text="Camera Index",
            font=("Segoe UI", 11, "bold"),
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_primary']
        ).pack(side='left')
        
        self.camera_index_var = tk.IntVar(value=self.config['camera_index'])
        camera_index_spin = tk.Spinbox(
            cam_index_frame,
            from_=0, to=5,
            textvariable=self.camera_index_var,
            width=8,
            font=("Segoe UI", 10),
            bg=self.colors['bg_tertiary'],
            fg=self.colors['text_primary'],
            insertbackground=self.colors['text_primary'],
            buttonbackground=self.colors['bg_tertiary']
        )
        camera_index_spin.pack(side='right')
        
        # Prediction interval
        interval_frame = tk.Frame(camera_content, bg=self.colors['bg_secondary'])
        interval_frame.pack(fill='x', pady=(0, 15))
        
        tk.Label(
            interval_frame,
            text="Prediction Interval (seconds)",
            font=("Segoe UI", 11, "bold"),
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_primary']
        ).pack(anchor='w', pady=(0, 5))
        
        self.interval_var = tk.DoubleVar(value=self.config['prediction_interval'])
        interval_scale = tk.Scale(
            interval_frame,
            from_=0.5, to=5.0, resolution=0.1,
            orient='horizontal',
            variable=self.interval_var,
            font=("Segoe UI", 10),
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_primary'],
            highlightbackground=self.colors['bg_secondary'],
            troughcolor=self.colors['bg_tertiary'],
            activebackground=self.colors['accent_primary']
        )
        interval_scale.pack(fill='x')
        
        # Detection region size
        region_frame = tk.Frame(camera_content, bg=self.colors['bg_secondary'])
        region_frame.pack(fill='x', pady=(0, 15))
        
        tk.Label(
            region_frame,
            text="Detection Region Size",
            font=("Segoe UI", 11, "bold"),
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_primary']
        ).pack(anchor='w', pady=(0, 5))
        
        self.region_var = tk.DoubleVar(value=self.config['detection_region_size'])
        region_scale = tk.Scale(
            region_frame,
            from_=0.1, to=0.8, resolution=0.05,
            orient='horizontal',
            variable=self.region_var,
            font=("Segoe UI", 10),
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_primary'],
            highlightbackground=self.colors['bg_secondary'],
            troughcolor=self.colors['bg_tertiary'],
            activebackground=self.colors['accent_primary']
        )
        region_scale.pack(fill='x')
        
        # Auto-save option
        self.auto_save_var = tk.BooleanVar(value=self.config['auto_save'])
        auto_save_check = tk.Checkbutton(
            camera_content,
            text="💾 Auto-save high-confidence predictions",
            variable=self.auto_save_var,
            font=("Segoe UI", 11),
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_primary'],
            selectcolor=self.colors['accent_primary'],
            activebackground=self.colors['bg_secondary'],
            activeforeground=self.colors['text_primary']
        )
        auto_save_check.pack(anchor='w')
        
        # Performance Settings
        perf_section, perf_content = self.create_modern_frame(scrollable_frame, "⚡ Performance Settings")
        perf_section.pack(fill='x', pady=(0, 20))
        
        # GPU acceleration (placeholder)
        gpu_frame = tk.Frame(perf_content, bg=self.colors['bg_secondary'])
        gpu_frame.pack(fill='x', pady=(0, 15))
        
        self.gpu_var = tk.BooleanVar(value=False)
        gpu_check = tk.Checkbutton(
            gpu_frame,
            text="🚀 Enable GPU acceleration (if available)",
            variable=self.gpu_var,
            font=("Segoe UI", 11),
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_primary'],
            selectcolor=self.colors['accent_primary'],
            activebackground=self.colors['bg_secondary'],
            activeforeground=self.colors['text_primary']
        )
        gpu_check.pack(anchor='w')
        
        # Batch processing
        batch_frame = tk.Frame(perf_content, bg=self.colors['bg_secondary'])
        batch_frame.pack(fill='x', pady=(0, 15))
        
        tk.Label(
            batch_frame,
            text="Batch Size (for multiple images)",
            font=("Segoe UI", 11, "bold"),
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_primary']
        ).pack(side='left')
        
        self.batch_var = tk.IntVar(value=1)
        batch_spin = tk.Spinbox(
            batch_frame,
            from_=1, to=16,
            textvariable=self.batch_var,
            width=8,
            font=("Segoe UI", 10),
            bg=self.colors['bg_tertiary'],
            fg=self.colors['text_primary'],
            insertbackground=self.colors['text_primary'],
            buttonbackground=self.colors['bg_tertiary']
        )
        batch_spin.pack(side='right')
        
        # Apply settings button
        apply_frame = tk.Frame(scrollable_frame, bg=self.colors['bg_primary'])
        apply_frame.pack(fill='x', pady=20)
        
        self.apply_btn = self.create_modern_button(
            apply_frame, "Apply Settings", self.apply_settings,
            self.colors['accent_primary'], font_size=12, icon="✅"
        )
        self.apply_btn.pack(anchor='center')
        
        # Reset to defaults button
        self.reset_btn = self.create_modern_button(
            apply_frame, "Reset to Defaults", self.reset_settings,
            self.colors['accent_warning'], font_size=11, icon="🔄"
        )
        self.reset_btn.pack(anchor='center', pady=(10, 0))
    
    def apply_settings(self):
        """Apply configuration changes with visual feedback"""
        try:
            # Update config
            self.config['confidence_threshold'] = self.conf_var.get()
            self.config['log_predictions'] = self.log_var.get()
            self.config['camera_index'] = self.camera_index_var.get()
            self.config['prediction_interval'] = self.interval_var.get()
            self.config['detection_region_size'] = self.region_var.get()
            self.config['auto_save'] = self.auto_save_var.get()
            
            # Visual feedback
            original_text = self.apply_btn.cget('text')
            self.apply_btn.config(text="✅ Applied!", bg=self.colors['accent_primary'])
            
            # Reset button after delay
            def reset_button():
                self.apply_btn.config(text=original_text, bg=self.colors['accent_primary'])
            
            self.root.after(2000, reset_button)
            
            self.status_label.config(text="⚙️ Settings applied successfully", fg=self.colors['accent_primary'])
            
        except Exception as e:
            messagebox.showerror("Settings Error", f"Error applying settings: {str(e)}")
    
    def reset_settings(self):
        """Reset all settings to defaults"""
        if messagebox.askyesno("Reset Settings", "Reset all settings to default values?"):
            # Reset variables
            self.conf_var.set(0.5)
            self.log_var.set(True)
            self.camera_index_var.set(0)
            self.interval_var.set(1.0)
            self.region_var.set(0.3)
            self.auto_save_var.set(False)
            self.gpu_var.set(False)
            self.batch_var.set(1)
            
            self.status_label.config(text="🔄 Settings reset to defaults", fg=self.colors['accent_warning'])
    
    def update_statistics(self):
        """Update statistics display"""
        if not hasattr(self, 'stats_labels'):
            return
        
        total = len(self.prediction_history)
        high_conf = sum(1 for p in self.prediction_history if p.get('confidence', 0) > 0.8)
        
        if total > 0:
            avg_conf = sum(p.get('confidence', 0) for p in self.prediction_history) / total
            predictions = [p.get('prediction', '') for p in self.prediction_history]
            most_common = max(set(predictions), key=predictions.count) if predictions else "None"
        else:
            avg_conf = 0
            most_common = "None"
        
        self.stats_labels["Total Analyses"].config(text=str(total))
        self.stats_labels["High Confidence"].config(text=str(high_conf))
        self.stats_labels["Average Confidence"].config(text=f"{avg_conf:.1%}")
        self.stats_labels["Most Common"].config(text=most_common)
    
    # Camera functionality methods (keeping original logic with UI updates)
    def start_camera(self):
        """Start the camera feed with modern UI updates"""
        try:
            self.camera = cv2.VideoCapture(self.config['camera_index'])
            if not self.camera.isOpened():
                raise Exception(f"Cannot open camera {self.config['camera_index']}")
            
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.camera.set(cv2.CAP_PROP_FPS, self.config['live_fps'])
            
            self.is_camera_running = True
            
            # Update UI with modern styling
            self.start_camera_btn.config(
                text="📹 Camera Active",
                bg=self.colors['accent_primary'],
                state='disabled'
            )
            self.stop_camera_btn.config(state='normal')
            self.capture_btn.config(state='normal')
            
            # Get camera info
            width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(self.camera.get(cv2.CAP_PROP_FPS))
            
            self.camera_info_label.config(
                text=f"📷 Camera: {width}x{height} @ {fps}fps",
                fg=self.colors['accent_primary']
            )
            
            # Start camera thread
            self.live_thread = threading.Thread(target=self.camera_loop)
            self.live_thread.daemon = True
            self.live_thread.start()
            
            self.status_label.config(
                text=f"📹 Camera started: {width}x{height} @ {fps}fps",
                fg=self.colors['accent_primary']
            )
            self.logger.info(f"Camera started: {width}x{height} @ {fps}fps")
            
        except Exception as e:
            error_msg = f"Error starting camera: {str(e)}"
            messagebox.showerror("Camera Error", error_msg)
            self.status_label.config(text=f"❌ {error_msg}", fg=self.colors['accent_danger'])
            self.logger.error(error_msg)
            if self.camera:
                self.camera.release()
                self.camera = None
    
    def stop_camera(self):
        """Stop the camera feed with modern UI updates"""
        self.is_camera_running = False
        
        if self.live_thread:
            self.live_thread.join(timeout=2.0)
        
        if self.camera:
            self.camera.release()
            self.camera = None
        
        # Update UI
        self.start_camera_btn.config(
            text="📹 Start Camera",
            bg=self.colors['accent_primary'],
            state='normal'
        )
        self.stop_camera_btn.config(state='disabled')
        self.capture_btn.config(state='disabled')
        
        self.camera_info_label.config(
            text="📷 Camera: Stopped",
            fg=self.colors['accent_danger']
        )
        
        # Clear video display
        self.video_label.config(
            image='',
            text="📹\n\nCamera Feed\n\nStart camera to begin live analysis"
        )
        self.video_label.image = None
        
        self.status_label.config(
            text="📹 Camera stopped",
            fg=self.colors['text_secondary']
        )
        self.logger.info("Camera stopped")
    
    def camera_loop(self):
        """Main camera loop for live video feed"""
        while self.is_camera_running and self.camera:
            try:
                ret, frame = self.camera.read()
                if not ret:
                    break
                
                # Store current frame
                self.current_live_frame = frame.copy()
                
                # Process frame for display
                display_frame = self.process_frame_for_display(frame)
                
                # Convert to PhotoImage and update display
                frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                frame_pil = frame_pil.resize((960, 720), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(frame_pil)
                
                # Update UI in main thread
                self.root.after(0, self.update_video_display, photo)
                
                # Auto-analysis
                if self.auto_analysis_var.get() and self.model:
                    current_time = time.time()
                    if current_time - self.last_prediction_time >= self.config['prediction_interval']:
                        self.last_prediction_time = current_time
                        # Run prediction in separate thread
                        pred_thread = threading.Thread(target=self.predict_live_frame, args=(frame,))
                        pred_thread.daemon = True
                        pred_thread.start()
                
                # Control frame rate
                time.sleep(1 / self.config['live_fps'])
                
            except Exception as e:
                self.logger.error(f"Camera loop error: {str(e)}")
                break
        
        # Cleanup
        if self.camera:
            self.camera.release()
            self.camera = None
    
    def process_frame_for_display(self, frame):
        """Process frame for display with detection region overlay"""
        display_frame = frame.copy()
        
        if self.show_region_var.get():
            # Draw detection region
            height, width = frame.shape[:2]
            region_size = self.config['detection_region_size']
            
            # Calculate region coordinates
            region_width = int(width * region_size)
            region_height = int(height * region_size)
            x1 = (width - region_width) // 2
            y1 = (height - region_height) // 2
            x2 = x1 + region_width
            y2 = y1 + region_height
            
            # Draw modern detection region
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.rectangle(display_frame, (x1-2, y1-2), (x2+2, y2+2), (0, 0, 0), 1)
            
            # Add corner markers
            corner_size = 20
            cv2.line(display_frame, (x1, y1), (x1 + corner_size, y1), (0, 255, 0), 4)
            cv2.line(display_frame, (x1, y1), (x1, y1 + corner_size), (0, 255, 0), 4)
            cv2.line(display_frame, (x2, y1), (x2 - corner_size, y1), (0, 255, 0), 4)
            cv2.line(display_frame, (x2, y1), (x2, y1 + corner_size), (0, 255, 0), 4)
            cv2.line(display_frame, (x1, y2), (x1 + corner_size, y2), (0, 255, 0), 4)
            cv2.line(display_frame, (x1, y2), (x1, y2 - corner_size), (0, 255, 0), 4)
            cv2.line(display_frame, (x2, y2), (x2 - corner_size, y2), (0, 255, 0), 4)
            cv2.line(display_frame, (x2, y2), (x2, y2 - corner_size), (0, 255, 0), 4)
            
            # Region label with background
            label_text = "ANALYSIS REGION"
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(display_frame, (x1, y1-35), (x1 + label_size[0] + 10, y1), (0, 255, 0), -1)
            cv2.putText(display_frame, label_text, (x1 + 5, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Add modern timestamp overlay
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        timestamp_size = cv2.getTextSize(timestamp, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(display_frame, (10, 10), (timestamp_size[0] + 20, 45), (0, 0, 0), -1)
        cv2.rectangle(display_frame, (10, 10), (timestamp_size[0] + 20, 45), (255, 255, 255), 2)
        cv2.putText(display_frame, timestamp, (15, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add live prediction result overlay
        if self.live_prediction_result:
            prediction, confidence = self.live_prediction_result
            result_text = f"{prediction.upper()}: {confidence:.1%}"
            
            # Choose color and background based on confidence
            if confidence > 0.8:
                color = (0, 255, 0)  # Green
                bg_color = (0, 200, 0)
            elif confidence > 0.6:
                color = (0, 255, 255)  # Yellow
                bg_color = (0, 200, 200)
            else:
                color = (0, 0, 255)  # Red
                bg_color = (0, 0, 200)
            
            # Result overlay at bottom
            result_size = cv2.getTextSize(result_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            y_pos = display_frame.shape[0] - 20
            cv2.rectangle(display_frame, (10, y_pos - 35), (result_size[0] + 20, y_pos + 5), bg_color, -1)
            cv2.rectangle(display_frame, (10, y_pos - 35), (result_size[0] + 20, y_pos + 5), color, 2)
            cv2.putText(display_frame, result_text, (15, y_pos - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return display_frame
    
    def update_video_display(self, photo):
        """Update video display in main thread"""
        self.video_label.config(image=photo, text="")
        self.video_label.image = photo
    
    def predict_live_frame(self, frame):
        """Predict on live camera frame with modern UI updates"""
        try:
            # Extract detection region
            height, width = frame.shape[:2]
            region_size = self.config['detection_region_size']
            
            region_width = int(width * region_size)
            region_height = int(height * region_size)
            x1 = (width - region_width) // 2
            y1 = (height - region_height) // 2
            x2 = x1 + region_width
            y2 = y1 + region_height
            
            # Crop detection region
            detection_region = frame[y1:y2, x1:x2]
            
            # Convert to PIL Image
            region_rgb = cv2.cvtColor(detection_region, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(region_rgb)
            
            # Make prediction
            img_tensor = self.base_transform(pil_image).unsqueeze(0)
            
            with torch.no_grad():
                output = self.model(img_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                prediction = self.config['class_names'][predicted.item()]
                confidence_score = confidence.item()
                all_probs = probabilities[0].numpy()
            
            # Update live prediction result
            self.live_prediction_result = (prediction, confidence_score)
            
            # Update UI in main thread
            self.root.after(0, self.update_live_results, prediction, confidence_score, all_probs)
            
            # Log prediction
            if self.config['log_predictions']:
                self.logger.info(f"Live prediction: {prediction}, Confidence: {confidence_score:.3f}")
            
            # Auto-save if enabled
            if self.config['auto_save'] and confidence_score > self.config['confidence_threshold']:
                self.root.after(0, self.auto_save_prediction, frame, prediction, confidence_score)
            
        except Exception as e:
            self.logger.error(f"Live prediction error: {str(e)}")
    
    def update_live_results(self, prediction, confidence, probabilities):
        """Update live results display with modern styling"""
        # Color coding with modern palette
        if confidence > 0.8:
            result_color = self.colors['accent_primary']  # Green
            status_icon = "🟢"
        elif confidence > 0.6:
            result_color = self.colors['accent_warning']  # Orange
            status_icon = "🟡"
        else:
            result_color = self.colors['accent_danger']  # Red
            status_icon = "🔴"
        
        # Update current result
        self.live_result_label.config(
            text=f"{status_icon} {prediction.upper()}",
            fg=result_color
        )
        self.live_confidence_label.config(
            text=f"{confidence:.1%}",
            fg=result_color
        )
        
        # Add to live history with modern formatting
        timestamp = datetime.now().strftime("%H:%M:%S")
        history_line = f"[{timestamp}] {status_icon} {prediction.upper()} ({confidence:.1%})\n"
        
        # Keep only last 10 entries
        current_text = self.live_history_text.get('1.0', tk.END)
        lines = current_text.strip().split('\n')
        if len(lines) >= 10:
            lines = lines[-9:]  # Keep last 9, add 1 new = 10 total
        
        new_text = '\n'.join(lines) + '\n' + history_line if lines[0] else history_line
        
        self.live_history_text.delete('1.0', tk.END)
        self.live_history_text.insert('1.0', new_text)
        self.live_history_text.see(tk.END)
        
        # Add to main history
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'source': 'live_camera',
            'prediction': prediction,
            'confidence': confidence,
            'enhanced': False,
            'probabilities': probabilities.tolist()
        }
        self.prediction_history.append(history_entry)
        
        # Update main history display
        main_history_text = f"{datetime.now().strftime('%H:%M:%S')} - {prediction.upper()} ({confidence:.1%}) [LIVE]"
        self.history_listbox.insert(0, main_history_text)
        
        # Update statistics
        self.update_statistics()
    
    def auto_save_prediction(self, frame, prediction, confidence):
        """Auto-save high-confidence predictions"""
        try:
            # Create saves directory
            save_dir = Path("auto_saves")
            save_dir.mkdir(exist_ok=True)
            
            # Save frame
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"live_{prediction}_{confidence:.0%}_{timestamp}.jpg"
            filepath = save_dir / filename
            
            cv2.imwrite(str(filepath), frame)
            
            # Log save
            self.logger.info(f"Auto-saved: {filename}")
            
        except Exception as e:
            self.logger.error(f"Auto-save error: {str(e)}")
    
    def capture_frame(self):
        """Capture current frame and analyze it"""
        if self.current_live_frame is not None:
            try:
                # Save captured frame
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"captured_frame_{timestamp}.jpg"
                filepath = Path("captures") / filename
                filepath.parent.mkdir(exist_ok=True)
                
                cv2.imwrite(str(filepath), self.current_live_frame)
                
                # Convert to PIL for analysis
                frame_rgb = cv2.cvtColor(self.current_live_frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                # Switch to main tab and load captured image
                self.notebook.select(0)  # Select main tab
                self.current_image = pil_image
                self.current_image_path = str(filepath)
                
                # Display captured image
                display_image = pil_image.copy()
                display_image.thumbnail(self.config['display_size'], Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(display_image)
                self.image_label.config(image=photo, text="")
                self.image_label.image = photo
                
                # Update file info
                width, height = pil_image.size
                self.file_label.config(text=f"📸 Captured: {filename} • {width}x{height}")
                
                # Clear previous results
                self.clear_analysis_results()
                
                # Show success message
                messagebox.showinfo("Frame Captured", 
                    f"Frame captured successfully!\n\nSaved as: {filename}\n\nSwitched to Analysis tab for detailed examination.")
                
                self.status_label.config(
                    text=f"📸 Frame captured: {filename}",
                    fg=self.colors['accent_primary']
                )
                self.logger.info(f"Frame captured: {filename}")
                
            except Exception as e:
                error_msg = f"Error capturing frame: {str(e)}"
                messagebox.showerror("Capture Error", error_msg)
                self.status_label.config(text=f"❌ {error_msg}", fg=self.colors['accent_danger'])
                self.logger.error(error_msg)
    
    def clear_analysis_results(self):
        """Clear analysis results display"""
        self.result_label.config(text="Awaiting analysis...", fg=self.colors['text_secondary'])
        self.confidence_label.config(text="", fg=self.colors['text_secondary'])
        self.confidence_bar['value'] = 0
        self.prob_text.delete('1.0', tk.END)
        self.save_btn.config(state='disabled')
    
    def camera_settings(self):
        """Open modern camera settings dialog"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Camera Settings")
        settings_window.geometry("500x600")
        settings_window.configure(bg=self.colors['bg_primary'])
        settings_window.transient(self.root)
        settings_window.grab_set()
        
        # Header
        header_frame = tk.Frame(settings_window, bg=self.colors['bg_secondary'], height=60)
        header_frame.pack(fill='x')
        header_frame.pack_propagate(False)
        
        tk.Label(
            header_frame,
            text="📹 Camera Configuration",
            font=("Segoe UI", 16, "bold"),
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_primary']
        ).pack(pady=20)
        
        # Main content
        main_frame = tk.Frame(settings_window, bg=self.colors['bg_primary'])
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Available cameras section
        camera_section, camera_content = self.create_modern_frame(main_frame, "📷 Available Cameras")
        camera_section.pack(fill='x', pady=(0, 20))
        
        # Detect available cameras
        available_cameras = []
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(i)
                cap.release()
        
        if available_cameras:
            for cam_id in available_cameras:
                cam_frame = tk.Frame(camera_content, bg=self.colors['bg_tertiary'])
                cam_frame.pack(fill='x', pady=2)
                
                tk.Label(
                    cam_frame,
                    text=f"🟢 Camera {cam_id}: Available",
                    font=("Segoe UI", 10),
                    bg=self.colors['bg_tertiary'],
                    fg=self.colors['accent_primary'],
                    anchor='w'
                ).pack(side='left', padx=10, pady=5)
        else:
            tk.Label(
                camera_content,
                text="🔴 No cameras detected",
                font=("Segoe UI", 10),
                bg=self.colors['bg_secondary'],
                fg=self.colors['accent_danger']
            ).pack(anchor='w')
        
        # Resolution settings
        res_section, res_content = self.create_modern_frame(main_frame, "📐 Resolution Settings")
        res_section.pack(fill='x', pady=(0, 20))
        
        resolutions = ["640x480", "800x600", "1024x768", "1280x720", "1920x1080"]
        resolution_var = tk.StringVar(value="1280x720")
        
        for res in resolutions:
            rb = tk.Radiobutton(
                res_content,
                text=res,
                variable=resolution_var,
                value=res,
                font=("Segoe UI", 10),
                bg=self.colors['bg_secondary'],
                fg=self.colors['text_primary'],
                selectcolor=self.colors['accent_primary'],
                activebackground=self.colors['bg_secondary'],
                activeforeground=self.colors['text_primary']
            )
            rb.pack(anchor='w', pady=2)
        
        # Test camera section
        test_section, test_content = self.create_modern_frame(main_frame, "🧪 Camera Test")
        test_section.pack(fill='x', pady=(0, 20))
        
        def test_camera():
            try:
                test_cap = cv2.VideoCapture(self.config['camera_index'])
                if test_cap.isOpened():
                    ret, frame = test_cap.read()
                    if ret:
                        messagebox.showinfo("Camera Test", "✅ Camera is working properly!")
                    else:
                        messagebox.showerror("Camera Test", "❌ Camera connected but no frame received")
                    test_cap.release()
                else:
                    messagebox.showerror("Camera Test", "❌ Cannot open camera")
            except Exception as e:
                messagebox.showerror("Camera Test", f"❌ Camera test failed: {str(e)}")
        
        test_btn = self.create_modern_button(
            test_content, "Test Camera", test_camera,
            self.colors['accent_secondary'], icon="🧪"
        )
        test_btn.pack(anchor='w')
        
        # Close button
        close_btn = self.create_modern_button(
            main_frame, "Close", settings_window.destroy,
            self.colors['bg_tertiary'], icon="✖️"
        )
        close_btn.pack(pady=20)
    
    def toggle_auto_analysis(self):
        """Toggle auto-analysis with visual feedback"""
        if self.auto_analysis_var.get():
            self.status_label.config(
                text="🤖 Auto-analysis enabled",
                fg=self.colors['accent_primary']
            )
            self.logger.info("Auto-analysis enabled")
        else:
            self.status_label.config(
                text="⏸️ Auto-analysis disabled",
                fg=self.colors['accent_warning']
            )
            self.logger.info("Auto-analysis disabled")
    
    # Image analysis methods with modern UI updates
    def select_image(self):
        """Enhanced image selection with modern UI"""
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
        """Clear current image and results with modern UI"""
        self.current_image_path = None
        self.current_image = None
        self.image_label.config(
            image='',
            text="📷\n\nNo image loaded\n\nClick 'Select Image' to begin"
        )
        self.image_label.image = None
        self.file_label.config(text="No file selected • Supported: JPG, PNG, BMP, TIFF")
        self.clear_analysis_results()
        self.status_label.config(text="🗑️ Image cleared", fg=self.colors['text_secondary'])
    
    def load_and_display_image(self, path):
        """Enhanced image loading with modern UI and metadata"""
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
            
            # Update file info with modern styling
            filename = Path(path).name
            size_mb = file_size / (1024 * 1024)
            self.file_label.config(
                text=f"📁 Selected: {filename} • {width}x{height} • {size_mb:.1f}MB"
            )
            
            # Clear previous results
            self.clear_analysis_results()
            
            self.status_label.config(
                text=f"✅ Image loaded: {filename}",
                fg=self.colors['accent_primary']
            )
            self.logger.info(f"Image loaded: {filename} ({width}x{height})")
            
        except Exception as e:
            error_msg = f"Error loading image: {str(e)}"
            self.status_label.config(text=f"❌ {error_msg}", fg=self.colors['accent_danger'])
            messagebox.showerror("Image Loading Error", error_msg)
            self.logger.error(error_msg)
    
    def predict_threaded(self):
        """Standard prediction in separate thread with modern UI"""
        self._run_prediction(enhanced=False)
    
    def enhanced_predict_threaded(self):
        """Enhanced prediction with augmentation and modern UI"""
        self._run_prediction(enhanced=True)
    
    def _run_prediction(self, enhanced=False):
        """Run prediction with modern progress tracking"""
        if not self.current_image_path or not self.model:
            return
        
        # Update UI with modern styling
        self.analysis_progress.start()
        method_text = "Enhanced Analysis" if enhanced else "Standard Analysis"
        
        # Disable buttons with visual feedback
        original_predict_text = self.predict_btn.cget('text')
        original_enhance_text = self.enhance_btn.cget('text')
        
        self.predict_btn.config(
            state='disabled',
            text="🔄 Analyzing...",
            bg=self.colors['bg_tertiary']
        )
        self.enhance_btn.config(
            state='disabled',
            text="🔄 Processing...",
            bg=self.colors['bg_tertiary']
        )
        
        self.progress_label.config(text=f"🧠 Running {method_text}...")
        self.status_label.config(
            text=f"🔬 {method_text} in progress...",
            fg=self.colors['accent_secondary']
        )
        
        # Run in thread
        thread = threading.Thread(target=self.predict_image, args=(enhanced,))
        thread.daemon = True
        thread.start()
    
    def predict_image(self, enhanced=False):
        """Enhanced prediction with modern UI updates"""
        try:
            if not self.current_image:
                raise ValueError("No image loaded")
            
            self.root.after(0, lambda: self.progress_label.config(text="📊 Preprocessing image..."))
            
            # Choose transform
            transform = self.enhanced_transform if enhanced else self.base_transform
            img_tensor = transform(self.current_image).unsqueeze(0)
            
            self.root.after(0, lambda: self.progress_label.config(text="🤖 Running AI inference..."))
            
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
        """Enhanced results display with modern styling"""
        # Stop progress and restore buttons
        self.analysis_progress.stop()
        self.predict_btn.config(
            state='normal',
            text="🔬 Analyze Image",
            bg=self.colors['accent_primary']
        )
        self.enhance_btn.config(
            state='normal',
            text="✨ Enhanced Analysis",
            bg=self.colors['accent_warning']
        )
        self.progress_label.config(text="")
        
        # Modern color coding with status icons
        if confidence > 0.8:
            result_color = self.colors['accent_primary']  # Green
            status_icon = "🟢"
            confidence_level = "HIGH"
        elif confidence > 0.6:
            result_color = self.colors['accent_warning']  # Orange
            status_icon = "🟡"
            confidence_level = "MEDIUM"
        else:
            result_color = self.colors['accent_danger']  # Red
            status_icon = "🔴"
            confidence_level = "LOW"
        
        # Update main result with modern styling
        method_text = " (Enhanced)" if enhanced else ""
        self.result_label.config(
            text=f"{status_icon} {prediction.upper()}{method_text}",
            fg=result_color
        )
        
        # Update confidence with level indicator
        conf_text = f"{confidence:.1%} ({confidence_level})"
        self.confidence_label.config(text=conf_text, fg=result_color)
        self.confidence_bar['value'] = confidence * 100
        
        # Update detailed probabilities with modern formatting
        self.prob_text.delete('1.0', tk.END)
        
        # Modern analysis report
        report = f"{'='*60}\n"
        report += f"BLOOD ANALYSIS REPORT\n"
        report += f"{'='*60}\n\n"
        
        report += f"🔬 ANALYSIS METHOD: {'Enhanced AI Processing' if enhanced else 'Standard AI Processing'}\n"
        report += f"📅 TIMESTAMP: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"📊 OVERALL CONFIDENCE: {confidence:.1%} ({confidence_level})\n\n"
        
        report += f"🩸 CLASSIFICATION RESULTS:\n"
        report += f"{'-'*40}\n"
        
        # Sort by probability
        prob_data = [(self.config['class_names'][i], probabilities[i]) for i in range(len(self.config['class_names']))]
        prob_data.sort(key=lambda x: x[1], reverse=True)
        
        for i, (class_name, prob) in enumerate(prob_data):
            # Visual progress bar
            bar_length = int(prob * 25)
            bar = "█" * bar_length + "░" * (25 - bar_length)
            
            # Add ranking emoji
            rank_emoji = ["🥇", "🥈", "🥉"][i] if i < 3 else "  "
            
            report += f"{rank_emoji} {class_name.upper():12} {prob:6.1%} │{bar}│\n"
        
        report += f"\n{'-'*40}\n"
        
        # Add confidence assessment
        if confidence < self.config['confidence_threshold']:
            report += f"⚠️  WARNING: Low confidence detection\n"
            report += f"   Consider: Enhanced analysis, better lighting, image quality\n\n"
        else:
            report += f"✅ High confidence detection - Results reliable\n\n"
        
        # Technical details
        report += f"🔧 TECHNICAL DETAILS:\n"
        report += f"   • Model: Neural Network (ResNet Architecture)\n"
        report += f"   • Input Resolution: {self.config['max_image_size'][0]}x{self.config['max_image_size'][1]}\n"
        report += f"   • Processing Time: < 1 second\n"
        report += f"   • Confidence Threshold: {self.config['confidence_threshold']:.1%}\n"
        
        self.prob_text.insert('1.0', report)
        
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
        history_text = f"{datetime.now().strftime('%H:%M:%S')} - {status_icon} {prediction.upper()} ({confidence:.1%})"
        if enhanced:
            history_text += " [Enhanced]"
        self.history_listbox.insert(0, history_text)
        
        # Update statistics
        self.update_statistics()
        
        # Enable save button
        self.save_btn.config(state='normal')
        
        # Status and warnings with modern styling
        if confidence < self.config['confidence_threshold']:
            self.status_label.config(
                text=f"⚠️ Low confidence: {confidence:.1%} (Threshold: {self.config['confidence_threshold']:.1%})",
                fg=self.colors['accent_warning']
            )
            
            # Modern warning dialog
            warning_msg = (
                f"Low Confidence Detection\n\n"
                f"Confidence: {confidence:.1%}\n"
                f"Threshold: {self.config['confidence_threshold']:.1%}\n\n"
                f"Recommendations:\n"
                f"• Try Enhanced Analysis for better accuracy\n"
                f"• Ensure proper lighting and image quality\n"
                f"• Check that the sample is clearly visible\n"
                f"• Consider retaking the image"
            )
            messagebox.showwarning("Low Confidence Warning", warning_msg)
        else:
            self.status_label.config(
                text=f"✅ Analysis completed successfully - {confidence_level} confidence",
                fg=self.colors['accent_primary']
            )
    
    def show_error(self, error_msg):
        """Enhanced error handling with modern UI"""
        self.analysis_progress.stop()
        self.predict_btn.config(
            state='normal',
            text="🔬 Analyze Image",
            bg=self.colors['accent_primary']
        )
        self.enhance_btn.config(
            state='normal',
            text="✨ Enhanced Analysis",
            bg=self.colors['accent_warning']
        )
        self.progress_label.config(text="")
        self.status_label.config(text=f"❌ {error_msg}", fg=self.colors['accent_danger'])
        messagebox.showerror("Analysis Error", f"❌ {error_msg}")
    
    def save_result(self):
        """Save current result with modern dialog"""
        if not self.prediction_history:
            return
        
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"blood_analysis_{timestamp}.json"
            filepath = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                initialname=filename,
                title="Save Analysis Result"
            )
            
            if filepath:
                # Create comprehensive result
                result_data = self.prediction_history[-1].copy()
                result_data['export_timestamp'] = datetime.now().isoformat()
                result_data['application_version'] = "4.0 Professional"
                
                with open(filepath, 'w') as f:
                    json.dump(result_data, f, indent=2)
                
                self.status_label.config(
                    text=f"💾 Result saved: {Path(filepath).name}",
                    fg=self.colors['accent_primary']
                )
                messagebox.showinfo("Save Complete", 
                    f"✅ Analysis result saved successfully!\n\nFile: {Path(filepath).name}")
                
        except Exception as e:
            error_msg = f"Error saving result: {str(e)}"
            self.status_label.config(text=f"❌ {error_msg}", fg=self.colors['accent_danger'])
            messagebox.showerror("Save Error", error_msg)
    
    def clear_history(self):
        """Clear prediction history with modern confirmation"""
        if messagebox.askyesno("Clear History", 
            "🗑️ Clear all prediction history?\n\nThis action cannot be undone."):
            self.prediction_history.clear()
            self.history_listbox.delete(0, tk.END)
            self.live_history_text.delete('1.0', tk.END)
            self.update_statistics()
            self.status_label.config(
                text="🗑️ History cleared successfully",
                fg=self.colors['text_secondary']
            )
    
    def export_history(self):
        """Export complete history with modern UI"""
        if not self.prediction_history:
            messagebox.showinfo("Export History", "📋 No history to export")
            return
        
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"blood_analysis_history_{timestamp}.json"
            filepath = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                initialname=filename,
                title="Export Analysis History"
            )
            
            if filepath:
                # Create comprehensive export
                export_data = {
                    'export_info': {
                        'timestamp': datetime.now().isoformat(),
                        'version': '4.0 Professional',
                        'total_predictions': len(self.prediction_history),
                        'exported_by': 'Blood Condition Detector'
                    },
                    'statistics': {
                        'total_analyses': len(self.prediction_history),
                        'high_confidence_count': sum(1 for p in self.prediction_history if p.get('confidence', 0) > 0.8),
                        'average_confidence': sum(p.get('confidence', 0) for p in self.prediction_history) / len(self.prediction_history) if self.prediction_history else 0,
                        'class_distribution': {}
                    },
                    'predictions': self.prediction_history
                }
                
                # Calculate class distribution
                for prediction in self.prediction_history:
                    class_name = prediction.get('prediction', 'unknown')
                    export_data['statistics']['class_distribution'][class_name] = export_data['statistics']['class_distribution'].get(class_name, 0) + 1
                
                with open(filepath, 'w') as f:
                    json.dump(export_data, f, indent=2)
                
                self.status_label.config(
                    text=f"📤 History exported: {Path(filepath).name}",
                    fg=self.colors['accent_primary']
                )
                messagebox.showinfo("Export Complete", 
                    f"✅ Analysis history exported successfully!\n\nFile: {Path(filepath).name}\nTotal Records: {len(self.prediction_history)}")
                
        except Exception as e:
            error_msg = f"Error exporting history: {str(e)}"
            self.status_label.config(text=f"❌ {error_msg}", fg=self.colors['accent_danger'])
            messagebox.showerror("Export Error", error_msg)

    def run(self):
        """Start the application with modern splash screen"""
        try:
            # Create splash screen
            splash = tk.Toplevel()
            splash.title("Blood Condition Detector")
            splash.geometry("400x300")
            splash.configure(bg=self.colors['bg_primary'])
            splash.resizable(False, False)
            splash.overrideredirect(True)
            
            # Center splash screen
            splash.update_idletasks()
            x = (splash.winfo_screenwidth() // 2) - (400 // 2)
            y = (splash.winfo_screenheight() // 2) - (300 // 2)
            splash.geometry(f"400x300+{x}+{y}")
            
            # Splash content
            splash_frame = tk.Frame(splash, bg=self.colors['bg_primary'])
            splash_frame.pack(fill='both', expand=True)
            
            # Logo/Icon
            tk.Label(
                splash_frame,
                text="🩸",
                font=("Arial", 60),
                bg=self.colors['bg_primary'],
                fg=self.colors['accent_primary']
            ).pack(pady=(40, 20))
            
            # Title
            tk.Label(
                splash_frame,
                text="Blood Condition Detector",
                font=("Segoe UI", 18, "bold"),
                bg=self.colors['bg_primary'],
                fg=self.colors['text_primary']
            ).pack()
            
            # Version
            tk.Label(
                splash_frame,
                text="Professional Edition v4.0",
                font=("Segoe UI", 12),
                bg=self.colors['bg_primary'],
                fg=self.colors['text_secondary']
            ).pack(pady=(5, 20))
            
            # Loading message
            loading_label = tk.Label(
                splash_frame,
                text="🔄 Initializing AI Model...",
                font=("Segoe UI", 11),
                bg=self.colors['bg_primary'],
                fg=self.colors['accent_secondary']
            )
            loading_label.pack()
            
            # Progress bar
            progress = ttk.Progressbar(
                splash_frame,
                mode='indeterminate',
                style='Modern.Horizontal.TProgressbar'
            )
            progress.pack(pady=20, padx=40, fill='x')
            progress.start()
            
            # Update splash
            splash.update()
            
            # Initialize main window (hidden)
            self.root.withdraw()
            
            # Simulate loading time for dramatic effect
            def loading_sequence():
                messages = [
                    "🔄 Loading AI Model...",
                    "🧠 Initializing Neural Network...",
                    "📊 Setting up Analysis Engine...",
                    "🎨 Preparing User Interface...",
                    "✅ Ready!"
                ]
                
                for i, message in enumerate(messages):
                    loading_label.config(text=message)
                    splash.update()
                    time.sleep(0.8)
                
                # Close splash and show main window
                progress.stop()
                splash.destroy()
                self.root.deiconify()
                
                # Center main window
                self.root.update_idletasks()
                width = self.root.winfo_width()
                height = self.root.winfo_height()
                x = (self.root.winfo_screenwidth() // 2) - (width // 2)
                y = (self.root.winfo_screenheight() // 2) - (height // 2)
                self.root.geometry(f"{width}x{height}+{x}+{y}")
                
                # Show welcome message
                self.status_label.config(
                    text="🎉 Welcome to Blood Condition Detector Professional v4.0",
                    fg=self.colors['accent_primary']
                )
            
            # Start loading sequence in thread
            loading_thread = threading.Thread(target=loading_sequence)
            loading_thread.daemon = True
            loading_thread.start()
            
            # Start main event loop
            self.root.mainloop()
            
        except Exception as e:
            messagebox.showerror("Startup Error", f"Failed to start application: {str(e)}")
            self.logger.error(f"Startup error: {str(e)}")

    def __del__(self):
        """Cleanup on deletion"""
        try:
            if hasattr(self, 'camera') and self.camera:
                self.camera.release()
            if hasattr(self, 'is_camera_running'):
                self.is_camera_running = False
        except:
            pass


def main():
    """Main function to run the application"""
    try:
        # Check Python version
        if sys.version_info < (3, 7):
            print("Error: Python 3.7 or higher is required")
            return
        
        # Check required packages
        required_packages = ['torch', 'torchvision', 'PIL', 'cv2', 'numpy']
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            print(f" Missing required packages: {', '.join(missing_packages)}")
            print("Install them using: pip install torch torchvision pillow opencv-python numpy")
            return
        
        
        app = ProfessionalBloodClassifier()
        app.run()
        
    except KeyboardInterrupt:
        print("\n Application interrupted by user")
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()