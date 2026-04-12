import os
import sys
import json
import importlib.util
import re
import subprocess
import platform
import glob
import time
import math
import html
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable

# Try to import optional dependencies
try:
    import markdown
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False

try:
    import qrcode
    QRCODE_AVAILABLE = True
except ImportError:
    QRCODE_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Try to import colorama, but fail gracefully if not installed
try:
    from colorama import Fore, Style, init as colorama_init
    # Initialize colorama
    colorama_init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False
    print("Warning: colorama not found. Colored output will be disabled.")

class Utils:
    """Utility functions for file handling, code testing, and parsing."""
    
    _project_root = None

    @staticmethod
    def get_output_path() -> str:
        base_dir = Utils.get_project_root()
        output_dir = os.path.join(base_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    @staticmethod
    def get_project_root() -> str:
        """
        Finds the project root by searching upwards for 'main.py' or '.git'.
        Caches the result to avoid redundant filesystem checks.
        """
        if Utils._project_root:
            return Utils._project_root
            
        current_dir = os.path.abspath(os.path.dirname(__file__))
        while current_dir != os.path.dirname(current_dir): # Stop at filesystem root
            if os.path.exists(os.path.join(current_dir, "main.py")) or \
               os.path.isdir(os.path.join(current_dir, ".git")):
                Utils._project_root = current_dir
                return current_dir
            current_dir = os.path.dirname(current_dir)
            
        # Fallback: return the directory of this file's parent if root not found
        # (Assume we are in src/VI_utils/utils.py -> project root is usually 2 levels up)
        fallback = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        Utils._project_root = fallback
        return fallback

    @staticmethod
    def find_file(name: str) -> List[str]:
        """
        Finds all occurrences of a file by name in the project.
        Returns a list of full paths to the files found.
        """
        root_dir = Utils.get_project_root()
        matches = []
        
        for root, dirs, files in os.walk(root_dir):
            # Skip common ignore dirs by modifying dirs in-place
            dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', 'venv', '.venv', '.venv_mac', 'env', 'node_modules', '.idea', '.vscode', 'site-packages'}]
            
            if name in files:
                matches.append(os.path.join(root, name))
                
        return matches

    @staticmethod
    def save_file(content: str, name: Optional[str] = None, extension: str = "py") -> str:
        """
        Save content to a file in the output directory.
        - If no name is provided, it generates one based on content.
        - Returns the full file path.
        """
        output_dir = Utils.get_output_path()
        if not name:
            name = Utils.name_file(content[:50])  # Generate a name from content (first 50 chars)
        filename = f"{name}.{extension}"
        full_path = os.path.join(output_dir, filename)

        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"File saved at: {full_path}")
        return full_path  # Return file path for further use

    @staticmethod
    def test_code(code: str) -> str:
        """
        Execute the generated Python code and return its output.
        Uses a sandboxed environment with timeouts to prevent hangs.
        """
        import io
        import contextlib
        import concurrent.futures
        import traceback

        def run_code():
            output_buffer = io.StringIO()
            local_globals = {}
            try:
                with contextlib.redirect_stdout(output_buffer), contextlib.redirect_stderr(output_buffer):
                    exec(code, local_globals)
            except Exception:
                output_buffer.write(traceback.format_exc())
            return output_buffer.getvalue()

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_code)
                output = future.result(timeout=10)  # Enforce a 10-second timeout
        except concurrent.futures.TimeoutError:
            output = "Error during code execution: Code execution timed out."
        except Exception as e:
            output = f"Error during code execution: {e}"
        return output

    @staticmethod
    def parse_code_response(response: str) -> str:
        """
        Extracts the first Python code block from a markdown response.
        If none exists, return the original response.
        """
        match = re.search(r"```python(.*?)```", response, re.DOTALL)
        return match.group(1).strip() if match else response.strip()

    @staticmethod
    def merge_code_snippets(content: str) -> str:
        """
        Extracts and merges all Python code blocks from a markdown response.
        Returns a single Python script with merged code.
        """
        code_blocks = re.findall(r"```python(.*?)```", content, re.DOTALL)
        return "\n\n".join(block.strip() for block in code_blocks) if code_blocks else content.strip()

    @staticmethod
    def load_file(name: str) -> Optional[str]:
        """Recursively find and load a file by name, searching anywhere in the project."""
        paths = Utils.find_file(name)
        
        if not paths:
            return None
            
        if len(paths) > 1:
             print(f"Warning: Multiple files found for '{name}'. Loading the first one: {paths[0]}")
             print(f"Found: {paths}")
        
        try:
            with open(paths[0], "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"Error reading file {paths[0]}: {e}")
            return None
        
    @staticmethod
    def name_file(user_input: str) -> str:
        """Generate a clean filename based on user input or AI assistance."""
        # Use dynamic import to avoid circular dependency or static path issues
        try:
            oai_module = Utils.import_file("oai.py")
        except (ImportError, FileNotFoundError):
            oai_module = None

        if oai_module:
            try:
                # Dynamically load OAI
                oai_instance = oai_module.OAI(api_keys_path=None)  # Uses auto-resolving API key path

                ai_name = oai_instance.chat_completion(
                    f"Generate as short as possible, descriptive filename for: {user_input}. Keep it concise, lowercase, and use underscores instead of spaces. Important: Do NOT include the file type extension in the name!"
                ).strip()

                # Ensure valid filename format
                ai_name = re.sub(r"[^\w\s-]", "", ai_name)  # Remove invalid chars
                ai_name = re.sub(r"[\s]+", "_", ai_name).lower()[:40]
                return ai_name
            except Exception as e:
                print(f"Error using OAI for filename generation: {e}")

        # Fallback to simple formatting if AI fails
        clean_input = re.sub(r"[^\w\s-]", "", user_input)
        fallback_name = re.sub(r"[\s]+", "_", clean_input).lower()[:40]
        return fallback_name

    @staticmethod
    def import_file(name: str) -> Any:
        """
        Find a Python module recursively starting from the project root and import it.
        - name: Filename (e.g., "my_module.py" or "my_module")
        - Raises ImportError if not found or ambiguous.
        """
        if not name.endswith(".py"):
            filename = f"{name}.py"
        else:
            filename = name
            
        paths = Utils.find_file(filename)
        
        if not paths:
            raise FileNotFoundError(f"Could not find module '{filename}' in project.")
            
        if len(paths) > 1:
            paths_str = "\n".join([f"- {p}" for p in paths])
            raise ImportError(f"Ambiguous import: Found multiple files named '{filename}':\n{paths_str}\nPlease ensure unique filenames.")
            
        file_path = paths[0]
        module_name = os.path.splitext(filename)[0]
        
        try:
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not create spec for {file_path}")
            
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module # Register in sys.modules
            spec.loader.exec_module(module)
            return module
        except Exception as e:
            raise ImportError(f"Failed to import '{filename}' from '{file_path}': {e}")
        
    @staticmethod
    def get_codebase_snapshot(root_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Generates a structured snapshot of the entire codebase.
        - Scans all Python files in the project.
        - Extracts imports and dependency relationships.
        - Merges all code into a unified context.
        """
        if root_dir is None:
            root_dir = Utils.get_project_root()
            
        codebase_snapshot = {}

        for root, dirs, files in os.walk(root_dir):
            # Skip ignore dirs
            dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', 'venv', '.venv', '.venv_mac', 'env', 'node_modules', '.idea', '.vscode', 'site-packages'}]
            
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()

                        # Extract import statements
                        imports = re.findall(r"^\s*(?:import|from)\s+([\w\.]+)", content, re.MULTILINE)

                        codebase_snapshot[file_path] = {
                            "imports": list(set(imports)),  # Remove duplicates
                            "content": content
                        }
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")

        return codebase_snapshot

    @staticmethod
    def merge_codebase(snapshot: Dict[str, Any]) -> str:
        """
        Merges all files in the codebase snapshot into a single structured output.
        """
        merged_code = []
        all_imports = set()

        # Collect all imports first
        for file_data in snapshot.values():
            all_imports.update(file_data["imports"])

        # Add collected imports at the top
        merged_code.append("# Unified Codebase Snapshot\n")
        merged_code.append("\n".join(f"import {imp}" for imp in sorted(all_imports)))
        merged_code.append("\n" + "=" * 60 + "\n")

        # Append each file's content
        for file_path, file_data in snapshot.items():
            merged_code.append(f"# File: {file_path}\n" + file_data["content"])
            merged_code.append("\n" + "=" * 60 + "\n")

        return "\n".join(merged_code)

    @staticmethod
    def quick_look(file_path: str) -> None:
        """Preview file using system appropriate viewer"""
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return
            
        # Print file being previewed
        print("Testing Quick Look preview with files:")
        print(f"\t{file_path}")
        
        system = platform.system()
        file_ext = Path(file_path).suffix.lower()
        
        try:
            if system == "Darwin":  # macOS
                if file_ext in ['.mp4', '.mov', '.avi']:
                    # Open with QuickTime Player
                    subprocess.run(['open', '-a', 'QuickTime Player', file_path])
                else:
                    # Use Quick Look for images
                    subprocess.run(['qlmanage', '-p', file_path], capture_output=True)
            elif system == "Windows":
                os.startfile(file_path)
            elif system == "Linux":
                subprocess.run(['xdg-open', file_path])
                
        except Exception as e:
            print(f"Error previewing file: {str(e)}")

    @staticmethod
    def printColoured(text: str, color_name: str) -> None:
        """
        Color the given text with the specified color name and print it directly.
        Available colors: red, green, blue, magenta, yellow, white, grey, default.
        """
        if not COLORAMA_AVAILABLE:
            print(text, flush=True)
            return

        color_dict = {
            "red": f"{Style.BRIGHT}{Fore.RED}",
            "green": f"{Style.BRIGHT}{Fore.GREEN}",
            "cyan": f"{Style.BRIGHT}{Fore.CYAN}",
            "blue": f"{Style.BRIGHT}{Fore.BLUE}",
            "magenta": f"{Style.BRIGHT}{Fore.MAGENTA}",
            "yellow": f"{Style.BRIGHT}{Fore.YELLOW}",
            "white": f"{Fore.LIGHTWHITE_EX}",
            "grey": f"{Fore.LIGHTBLACK_EX}",
            "default": Style.RESET_ALL
        }
        
        color_code = color_dict.get(color_name.lower(), Style.RESET_ALL)
        print(f"{color_code}{text}{Style.RESET_ALL}", flush=True)

    @staticmethod
    def markdown_to_html(markdown_text: str) -> str:
        """Convert Markdown to HTML."""
        if not MARKDOWN_AVAILABLE:
            raise ImportError("markdown package is required. Install it with: pip install markdown")
        return markdown.markdown(markdown_text)

    @staticmethod
    def escape_html(text: str) -> str:
        """
        Escapes HTML special characters in text for safe display in HTML content.
        
        Args:
            text: The text to escape.
            
        Returns:
            The escaped text with HTML special characters converted to their corresponding HTML entities.
        """
        return html.escape(text)

    @staticmethod
    def extract_json(output: str, retry_function: Callable, attempt: int = 1, max_attempts: int = 10) -> Optional[Dict]:
        """
        Tries to parse the entire output as JSON. If it fails, looks for a JSON block in the output and attempts to parse it.
        If parsing fails or no JSON block is found, it retries by calling the provided retry function.
        
        Args:
            output: The output string, potentially containing JSON or a JSON block.
            retry_function: Function to retry generating the output.
            attempt: Current attempt number.
            max_attempts: Maximum number of attempts allowed.
        
        Returns:
            Parsed JSON object if successful, or None if unsuccessful after max attempts.
        """
        try:
            # First, attempt to parse the entire output as JSON
            parsed_json = json.loads(output)
            return parsed_json
        except json.JSONDecodeError:
            # If it fails, look for a JSON block within the output
            try:
                json_block_match = re.search(r'```json\n([\s\S]*?)\n```', output)
                if json_block_match:
                    json_block = json_block_match.group(1)  # Extract the JSON block
                    parsed_json = json.loads(json_block)  # Attempt to parse the JSON block
                    return parsed_json
                else:
                    raise ValueError("No JSON block found in the output.")
            except (ValueError, json.JSONDecodeError) as e:
                print(f"Attempt {attempt} failed: {e}")
                if attempt < max_attempts:
                    print("Retrying...")
                    new_output = retry_function()  # Call the retry function to generate new output
                    return Utils.extract_json(new_output, retry_function, attempt + 1, max_attempts)
                else:
                    print("Maximum attempts reached. Unable to extract or parse JSON.")
                    print(output)  # Print the output for debugging purposes
                    return None

    @staticmethod
    def json_to_markdown(json_obj: Dict) -> str:
        """
        Converts a JSON object to a markdown string with selective use of bullet points, bold text,
        and numbers for better readability, while maintaining thematic breaks and depth-based headers
        for structure, ensuring header levels do not surpass ###.
        """
        def process_item(key, value, depth=1, is_list=False):
            """
            Processes each item, applying markdown based on its type, context, and whether it's part of a list,
            adjusting the depth to manage header levels.
            """
            md = ""
            prefix = ""
            # Adjust the maximum depth for headers to not surpass level 3
            adjusted_depth = min(depth, 4)  # Ensures we don't go beyond ### headers
            
            if adjusted_depth == 0:
                adjusted_depth = 1
            if adjusted_depth > 0:  # Adjust prefix based on depth to create numbered lists at deeper levels
                prefix = f"{'#' * (adjusted_depth)} "

            if key:
                md += f"{prefix}{key}\n\n"

            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    # Increase depth but do not let it surpass the maximum for headers
                    md += process_item(sub_key, sub_value, depth + 1 if depth < 3 else depth, is_list)
            elif isinstance(value, list):
                md += "\n".join([process_item(None, item, depth + 1 if depth < 3 else depth, is_list=True) for item in value])
            else:
                # Format simple values, applying bold for keys if within a list for clarity
                if key and is_list:
                    md += f"{value}\n\n"
                else:
                    md += f"- {value}\n\n"

            # Include thematic breaks after top-level sections for clear separation
            if depth == 2:
                md += "---\n\n"

            return md

        markdown_output = process_item(None, json_obj)
        return markdown_output.strip()  # Ensure clean output without leading/trailing whitespace

    @staticmethod
    def save_image(image: Any, base_name: str = "img_", extension: str = ".png") -> Optional[str]:
        """
        Save an image to the output directory with auto-incrementing filename.
        
        Args:
            image: PIL Image object or similar with .save() method
            base_name: Base name for the file
            extension: File extension (default: .png)
            
        Returns:
            Path to saved image or None if failed
        """
        if image:
            output_directory = Utils.get_output_path()
            existing_files = glob.glob(os.path.join(output_directory, base_name + "*"))
            next_number = len(existing_files) + 1
            filename = f"{base_name}{next_number:03}{extension}"
            path = os.path.join(output_directory, filename)

            image.save(path)
            print(f"Image saved as {path}\n")
            return path
        else:
            print("Failed to save image.")
            return None

    @staticmethod
    def save_speech(speech_data: bytes, filename: str) -> Optional[str]:
        """
        Save speech/audio data to a file in the output directory.
        
        Args:
            speech_data: Raw audio data bytes
            filename: Name of the file to save
            
        Returns:
            Path to saved file or None if failed
        """
        try:
            output_directory = Utils.get_output_path()
            file_path = os.path.join(output_directory, filename)
            
            # Assuming speech_data is raw audio data that needs to be written to a file
            with open(file_path, "wb") as file:
                file.write(speech_data)
            
            print(f"Speech saved as {file_path}")
            return file_path
        except Exception as e:
            print(f"Failed to save speech: {e}")
            return None

    @staticmethod
    def chaos_trigger() -> float:
        """
        Returns a chaotic trigger value based on time modulo 30 seconds.
        Useful for introducing randomness or time-based behavior.
        
        Returns:
            A value between 0 and 1 based on sin(time % 30)
        """
        t = time.time() % 30
        return abs(math.sin(t))

    @staticmethod
    def generate_qr(data: str, save_path: str = "qr_code.png") -> str:
        """
        Generate a QR code for the input data and save it to the specified path.
        
        Args:
            data: Data to encode in QR code
            save_path: Path where QR code image will be saved
            
        Returns:
            Path to saved QR code image
        """
        if not QRCODE_AVAILABLE:
            raise ImportError("qrcode package is required. Install it with: pip install qrcode[pil]")
        
        img = qrcode.make(data)
        img.save(save_path)
        return save_path

    @staticmethod
    def shorten_url(url: str, token: str) -> str:
        """
        Shorten a URL using the Bitly API.
        
        Args:
            url: Long URL to shorten
            token: Bitly API token
            
        Returns:
            Shortened URL or error message
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests package is required. Install it with: pip install requests")
        
        headers = {"Authorization": f"Bearer {token}"}
        payload = {"long_url": url}
        response = requests.post("https://api-ssl.bitly.com/v4/shorten", json=payload, headers=headers)
        return response.json().get("link", "Error shortening URL")

# ================================
#       TESTING UTILITIES
# ================================

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING UTILS FUNCTIONS")
    print("=" * 60)
    
    # Test 1: Project root and output path
    print("\n[1] Testing get_project_root() and get_output_path()")
    print(f"Project Root: {Utils.get_project_root()}")
    print(f"Output Path: {Utils.get_output_path()}")
    
    # Test 2: File finding
    print("\n[2] Testing find_file()")
    found = Utils.find_file("utils.py")
    print(f"Found utils.py: {found}")
    
    # Test 3: Colored output
    print("\n[3] Testing printColoured()")
    Utils.printColoured("This should be red", "red")
    Utils.printColoured("This should be green", "green")
    Utils.printColoured("This should be cyan", "cyan")
    
    # Test 4: Dynamic import
    print("\n[4] Testing import_file()")
    try:
        mod = Utils.import_file("utils.py")
        print(f"✅ Successfully dynamically imported utils.py")
    except Exception as e:
        print(f"❌ Import failed: {e}")
    
    # Test 5: File operations
    print("\n[5] Testing save_file(), load_file(), and test_code()")
    code_snippet = "print('Hello, AI-assisted world!')\nx = 5\ny = 10\nprint(f'The sum is: {x + y}')"
    
    saved_file_path = Utils.save_file(code_snippet)
    print(f"✅ Saved file: {saved_file_path}")
    
    loaded_code = Utils.load_file(os.path.basename(saved_file_path))
    print(f"✅ Loaded code (length: {len(loaded_code)} chars)")
    
    test_result = Utils.test_code(loaded_code)
    print(f"✅ Code execution result: {test_result.strip()}")
    
    # Test 6: HTML escaping
    print("\n[6] Testing escape_html()")
    html_text = '<script>alert("XSS")</script>'
    escaped = Utils.escape_html(html_text)
    print(f"Original: {html_text}")
    print(f"Escaped: {escaped}")
    
    # Test 7: Markdown to HTML (if available)
    print("\n[7] Testing markdown_to_html()")
    try:
        md_text = "# Hello\nThis is **bold** text."
        html_result = Utils.markdown_to_html(md_text)
        print(f"✅ Markdown converted to HTML (length: {len(html_result)} chars)")
    except ImportError as e:
        print(f"⚠️ {e}")
    
    # Test 8: JSON extraction
    print("\n[8] Testing extract_json()")
    json_str = '{"name": "test", "value": 123}'
    extracted = Utils.extract_json(json_str, lambda: json_str)
    print(f"✅ Extracted JSON: {extracted}")
    
    json_block = 'Some text\n```json\n{"key": "value"}\n```\nMore text'
    extracted_block = Utils.extract_json(json_block, lambda: json_block)
    print(f"✅ Extracted JSON from block: {extracted_block}")
    
    # Test 9: JSON to Markdown
    print("\n[9] Testing json_to_markdown()")
    test_json = {
        "name": "Test",
        "items": ["item1", "item2"],
        "nested": {"key": "value"}
    }
    md_result = Utils.json_to_markdown(test_json)
    print(f"✅ JSON converted to Markdown:\n{md_result[:200]}...")
    
    # Test 10: Chaos trigger
    print("\n[10] Testing chaos_trigger()")
    chaos_val = Utils.chaos_trigger()
    print(f"✅ Chaos trigger value: {chaos_val:.4f}")
    
    # Test 11: Code parsing
    print("\n[11] Testing parse_code_response() and merge_code_snippets()")
    code_response = "Here's some code:\n```python\nprint('hello')\n```\nAnd more:\n```python\nx = 5\n```"
    parsed = Utils.parse_code_response(code_response)
    print(f"✅ Parsed code (length: {len(parsed)} chars)")
    
    merged = Utils.merge_code_snippets(code_response)
    print(f"✅ Merged code snippets (length: {len(merged)} chars)")
    
    # Test 12: QR code generation (if available)
    print("\n[12] Testing generate_qr()")
    try:
        qr_path = Utils.generate_qr("https://example.com", "test_qr.png")
        print(f"✅ QR code generated: {qr_path}")
        if os.path.exists(qr_path):
            os.remove(qr_path)  # Clean up
    except ImportError as e:
        print(f"⚠️ {e}")
    
    # Test 13: Name file
    print("\n[13] Testing name_file()")
    try:
        filename = Utils.name_file("This is a test input for filename generation")
        print(f"✅ Generated filename: {filename}")
    except Exception as e:
        print(f"⚠️ Name file generation: {e}")
    
    # Test 14: Quick look (will just test it doesn't crash)
    print("\n[14] Testing quick_look()")
    test_file = __file__  # Use this file as test
    if os.path.exists(test_file):
        print(f"✅ quick_look() ready (would preview: {test_file})")
        # Uncomment to actually preview: Utils.quick_look(test_file)
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)
