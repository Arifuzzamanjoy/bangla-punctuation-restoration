#!/usr/bin/env python3
"""
Gradio interface for Bangla punctuation restoration
"""

import os
import sys
import logging
from typing import Optional, Tuple, List
import gradio as gr

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.baseline_model import PunctuationRestorer
from config import GRADIO_CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GradioInterface:
    """Gradio interface for punctuation restoration"""
    
    def __init__(self, model_path: Optional[str] = None, model_type: str = "token_classification"):
        """
        Initialize Gradio interface
        
        Args:
            model_path: Path to trained model
            model_type: Type of model to use
        """
        self.model_path = model_path
        self.model_type = model_type
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the punctuation restoration model"""
        try:
            if self.model_path and os.path.exists(self.model_path):
                logger.info(f"Loading model from: {self.model_path}")
                self.model = PunctuationRestorer(model_path=self.model_path, model_type=self.model_type)
                logger.info("Model loaded successfully")
            else:
                logger.warning("No valid model path provided")
                self.model = None
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = None
    
    def restore_punctuation(self, text: str) -> Tuple[str, str]:
        """
        Restore punctuation in text
        
        Args:
            text: Input text without punctuation
            
        Returns:
            Tuple of (status_message, punctuated_text)
        """
        if not text.strip():
            return "⚠️ Please enter some text", ""
        
        if len(text) > GRADIO_CONFIG["max_text_length"]:
            return f"⚠️ Text too long. Maximum {GRADIO_CONFIG['max_text_length']} characters allowed", ""
        
        if self.model is None or not self.model.model.trained:
            return "❌ Model not loaded or not trained", ""
        
        try:
            punctuated_text = self.model.restore_punctuation(text)
            return "✅ Punctuation restored successfully", punctuated_text
        except Exception as e:
            logger.error(f"Error during punctuation restoration: {e}")
            return f"❌ Error: {str(e)}", ""
    
    def batch_restore_punctuation(self, text_list: str) -> Tuple[str, str]:
        """
        Restore punctuation for multiple texts
        
        Args:
            text_list: Text with multiple sentences (one per line)
            
        Returns:
            Tuple of (status_message, results)
        """
        if not text_list.strip():
            return "⚠️ Please enter some text", ""
        
        if self.model is None or not self.model.model.trained:
            return "❌ Model not loaded or not trained", ""
        
        try:
            lines = [line.strip() for line in text_list.split('\n') if line.strip()]
            
            if len(lines) > 50:  # Limit batch size
                return "⚠️ Too many lines. Maximum 50 lines allowed", ""
            
            results = []
            for i, line in enumerate(lines, 1):
                try:
                    punctuated = self.model.restore_punctuation(line)
                    results.append(f"{i}. Original: {line}")
                    results.append(f"   Punctuated: {punctuated}")
                    results.append("")
                except Exception as e:
                    results.append(f"{i}. Original: {line}")
                    results.append(f"   Error: {str(e)}")
                    results.append("")
            
            return f"✅ Processed {len(lines)} sentences", "\n".join(results)
            
        except Exception as e:
            logger.error(f"Error during batch processing: {e}")
            return f"❌ Error: {str(e)}", ""
    
    def get_model_info(self) -> str:
        """Get information about the loaded model"""
        if self.model is None:
            return "❌ No model loaded"
        
        info = [
            "📊 Model Information:",
            f"• Type: {self.model.model.model_type}",
            f"• Trained: {'Yes' if self.model.model.trained else 'No'}",
            f"• Path: {self.model_path or 'Not specified'}"
        ]
        
        if hasattr(self.model.model, 'config'):
            config = self.model.model.config
            info.extend([
                f"• Model Name: {config.get('name', 'Unknown')}",
                f"• Max Length: {config.get('max_length', 'Unknown')}",
                f"• Batch Size: {config.get('batch_size', 'Unknown')}"
            ])
        
        return "\n".join(info)

def create_gradio_interface(model_path: Optional[str] = None, 
                          model_type: str = "token_classification") -> gr.Blocks:
    """
    Create Gradio interface
    
    Args:
        model_path: Path to trained model
        model_type: Type of model to use
        
    Returns:
        Gradio Blocks interface
    """
    interface = GradioInterface(model_path, model_type)
    
    with gr.Blocks(
        title="Bangla Punctuation Restoration",
        theme=gr.themes.Soft(),
        css="""
        .container { max-width: 1200px; margin: auto; }
        .header { text-align: center; margin-bottom: 2rem; }
        .example-box { background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; }
        """
    ) as demo:
        
        gr.HTML("""
        <div class="header">
            <h1>🔤 Bangla Punctuation Restoration</h1>
            <p>Restore punctuation in unpunctuated Bangla text using advanced machine learning models</p>
        </div>
        """)
        
        with gr.Tabs():
            # Single Text Tab
            with gr.TabItem("Single Text"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML("<h3>📝 Input</h3>")
                        input_text = gr.Textbox(
                            label="Unpunctuated Text",
                            placeholder="আমি তোমাকে বলেছিলাম তুমি কেন আসোনি আজ স্কুলে",
                            lines=5,
                            max_lines=10
                        )
                        
                        with gr.Row():
                            restore_btn = gr.Button("Restore Punctuation", variant="primary")
                            clear_btn = gr.Button("Clear", variant="secondary")
                    
                    with gr.Column(scale=1):
                        gr.HTML("<h3>✨ Output</h3>")
                        status_msg = gr.Textbox(label="Status", interactive=False)
                        output_text = gr.Textbox(
                            label="Punctuated Text",
                            lines=5,
                            max_lines=10,
                            interactive=False
                        )
                
                # Examples
                gr.HTML("<h3>💡 Examples</h3>")
                examples = gr.Examples(
                    examples=GRADIO_CONFIG["examples"],
                    inputs=[input_text],
                    label="Click on an example to try it:"
                )
            
            # Batch Processing Tab
            with gr.TabItem("Batch Processing"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML("<h3>📄 Batch Input</h3>")
                        batch_input = gr.Textbox(
                            label="Multiple Texts (one per line)",
                            placeholder="আমি ভালো আছি\nতুমি কেমন আছো\nআজ আবহাওয়া খুব ভালো",
                            lines=8,
                            max_lines=15
                        )
                        
                        with gr.Row():
                            batch_btn = gr.Button("Process Batch", variant="primary")
                            batch_clear_btn = gr.Button("Clear", variant="secondary")
                    
                    with gr.Column(scale=1):
                        gr.HTML("<h3>📋 Batch Results</h3>")
                        batch_status = gr.Textbox(label="Status", interactive=False)
                        batch_output = gr.Textbox(
                            label="Results",
                            lines=8,
                            max_lines=15,
                            interactive=False
                        )
            
            # Model Info Tab
            with gr.TabItem("Model Information"):
                with gr.Column():
                    gr.HTML("<h3>🤖 Model Details</h3>")
                    model_info = gr.Textbox(
                        label="Model Information",
                        value=interface.get_model_info(),
                        lines=10,
                        interactive=False
                    )
                    
                    refresh_info_btn = gr.Button("Refresh Info", variant="secondary")
                
                gr.HTML("""
                <div class="example-box">
                    <h4>📖 About This System</h4>
                    <p>This system uses advanced machine learning models to restore punctuation in Bangla text. 
                    It supports the following punctuation marks:</p>
                    <ul>
                        <li><strong>Comma (,)</strong> - For pauses and separating clauses</li>
                        <li><strong>Dari (।)</strong> - Bangla full stop</li>
                        <li><strong>Question mark (?)</strong> - For questions</li>
                        <li><strong>Exclamation mark (!)</strong> - For emphasis</li>
                        <li><strong>Semicolon (;)</strong> - For separating related clauses</li>
                        <li><strong>Colon (:)</strong> - For introducing lists or explanations</li>
                        <li><strong>Hyphen (-)</strong> - For compound words</li>
                    </ul>
                    <p>The system is trained on diverse Bangla text sources including literature, news, and online content.</p>
                </div>
                """)
        
        # Event handlers
        restore_btn.click(
            interface.restore_punctuation,
            inputs=[input_text],
            outputs=[status_msg, output_text]
        )
        
        clear_btn.click(
            lambda: ("", ""),
            outputs=[input_text, output_text]
        )
        
        batch_btn.click(
            interface.batch_restore_punctuation,
            inputs=[batch_input],
            outputs=[batch_status, batch_output]
        )
        
        batch_clear_btn.click(
            lambda: ("", ""),
            outputs=[batch_input, batch_output]
        )
        
        refresh_info_btn.click(
            interface.get_model_info,
            outputs=[model_info]
        )
        
        # Auto-refresh model info on load
        demo.load(
            interface.get_model_info,
            outputs=[model_info]
        )
    
    return demo

def launch_interface(model_path: Optional[str] = None,
                    model_type: str = "token_classification",
                    share: bool = None,
                    host: str = None,
                    port: int = None):
    """
    Launch Gradio interface
    
    Args:
        model_path: Path to trained model
        model_type: Type of model to use
        share: Whether to create public link
        host: Host address
        port: Port number
    """
    demo = create_gradio_interface(model_path, model_type)
    
    # Use config defaults if not provided
    share = share if share is not None else GRADIO_CONFIG["share"]
    host = host or GRADIO_CONFIG["host"]
    port = port or GRADIO_CONFIG["port"]
    
    logger.info(f"Launching Gradio interface on {host}:{port}")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Share: {share}")
    
    demo.launch(
        server_name=host,
        server_port=port,
        share=share,
        enable_queue=GRADIO_CONFIG["enable_queue"]
    )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Launch Gradio interface for punctuation restoration')
    parser.add_argument('--model_path', type=str, default="models/baseline",
                       help='Path to trained model')
    parser.add_argument('--model_type', type=str, default="token_classification",
                       choices=['token_classification', 'seq2seq'],
                       help='Type of model to use')
    parser.add_argument('--share', action='store_true',
                       help='Create public link')
    parser.add_argument('--host', type=str, default=None,
                       help='Host address')
    parser.add_argument('--port', type=int, default=None,
                       help='Port number')
    
    args = parser.parse_args()
    
    launch_interface(
        model_path=args.model_path,
        model_type=args.model_type,
        share=args.share,
        host=args.host,
        port=args.port
    )
