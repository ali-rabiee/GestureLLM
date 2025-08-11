import pygame
import sys
import time
from config import ACTIONS

class GUIController:
    def __init__(self):
        """Initialize pygame GUI controller"""
        pygame.init()
        
        # Display settings
        self.width = 550
        self.height = 350
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Robotic Arm GUI Controller")
        
        # Colors
        self.colors = {
            'bg': (240, 240, 240),
            'button': (70, 130, 180),
            'button_hover': (100, 149, 237),
            'button_pressed': (25, 25, 112),
            'text': (255, 255, 255),
            'text_dark': (50, 50, 50),
            'mode_button': (60, 179, 113),
            'mode_button_active': (34, 139, 34),
            'border': (200, 200, 200)
        }
        
        # Fonts
        self.font_large = pygame.font.Font(None, 32)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 18)
        
        # Control state
        self.mode = 0  # 0: Translation, 1: Orientation, 2: Gripper
        self.modes = ['Translation', 'Orientation', 'Gripper']
        self.current_action = -1
        self.pressed_buttons = set()
        
        # Create action mapping from config
        self.action_dict = dict(ACTIONS)
        
        # Define button layouts for each mode
        self.setup_button_layouts()
        
        # Clock for consistent frame rate
        self.clock = pygame.time.Clock()
        self.running = True

        # Prompt UI state (for shared autonomy suggestions)
        self.prompt = None  # dict with keys: type, text, goal_id
        self.prompt_buttons = {}
        self.prompt_response = None  # 'accept' | 'decline' | None
        self.prompt_overlay_height = 110
        self.prompt_margin = 10
        
    def setup_button_layouts(self):
        """Setup button layouts for different modes"""
        button_width = 100
        button_height = 40
        margin = 5
        
        # Translation mode buttons (actions 0-5)
        self.translation_buttons = {
            0: {'rect': pygame.Rect(225, 105, button_width, button_height), 'text': 'Forward'},
            1: {'rect': pygame.Rect(225, 175, button_width, button_height), 'text': 'Backward'},
            2: {'rect': pygame.Rect(115, 140, button_width, button_height), 'text': 'Left'},
            3: {'rect': pygame.Rect(335, 140, button_width, button_height), 'text': 'Right'},
            4: {'rect': pygame.Rect(225, 70, button_width, button_height), 'text': 'Up'},
            5: {'rect': pygame.Rect(225, 210, button_width, button_height), 'text': 'Down'}
        }
        
        # Orientation mode buttons (actions 6-11)
        self.orientation_buttons = {
            6: {'rect': pygame.Rect(115, 105, button_width, button_height), 'text': 'X+'},
            7: {'rect': pygame.Rect(115, 155, button_width, button_height), 'text': 'X-'},
            8: {'rect': pygame.Rect(225, 105, button_width, button_height), 'text': 'Y+'},
            9: {'rect': pygame.Rect(225, 155, button_width, button_height), 'text': 'Y-'},
            10: {'rect': pygame.Rect(335, 105, button_width, button_height), 'text': 'Z+'},
            11: {'rect': pygame.Rect(335, 155, button_width, button_height), 'text': 'Z-'}
        }
        
        # Gripper mode buttons (actions 12-13)
        self.gripper_buttons = {
            12: {'rect': pygame.Rect(170, 125, button_width, button_height), 'text': 'Open'},
            13: {'rect': pygame.Rect(280, 125, button_width, button_height), 'text': 'Close'}
        }
        
        # Mode switching buttons
        mode_button_width = 110
        mode_button_height = 30
        start_x = (self.width - (3 * mode_button_width + 2 * margin)) // 2
        
        self.mode_buttons = []
        for i, mode_name in enumerate(self.modes):
            button_rect = pygame.Rect(
                start_x + i * (mode_button_width + margin),
                35,
                mode_button_width,
                mode_button_height
            )
            self.mode_buttons.append({
                'rect': button_rect,
                'text': mode_name,
                'mode': i
            })
    
    def get_current_buttons(self):
        """Get buttons for current mode"""
        if self.mode == 0:
            return self.translation_buttons
        elif self.mode == 1:
            return self.orientation_buttons
        elif self.mode == 2:
            return self.gripper_buttons
        return {}
    
    def draw_button(self, button_info, action_id=None, is_pressed=False):
        """Draw a button with proper styling"""
        rect = button_info['rect']
        text = button_info['text']
        
        # Determine button color based on state
        if is_pressed:
            color = self.colors['button_pressed']
        else:
            mouse_pos = pygame.mouse.get_pos()
            if rect.collidepoint(mouse_pos):
                color = self.colors['button_hover']
            else:
                color = self.colors['button']
        
        # Draw button background
        pygame.draw.rect(self.screen, color, rect)
        pygame.draw.rect(self.screen, self.colors['border'], rect, 2)
        
        # Draw button text
        text_surface = self.font_medium.render(text, True, self.colors['text'])
        text_rect = text_surface.get_rect(center=rect.center)
        self.screen.blit(text_surface, text_rect)
    
    def draw_mode_button(self, button_info, is_active=False):
        """Draw a mode switching button"""
        rect = button_info['rect']
        text = button_info['text']
        
        # Determine button color
        if is_active:
            color = self.colors['mode_button_active']
        else:
            mouse_pos = pygame.mouse.get_pos()
            if rect.collidepoint(mouse_pos):
                color = self.colors['button_hover']
            else:
                color = self.colors['mode_button']
        
        # Draw button
        pygame.draw.rect(self.screen, color, rect)
        pygame.draw.rect(self.screen, self.colors['border'], rect, 2)
        
        # Draw text
        text_surface = self.font_medium.render(text, True, self.colors['text'])
        text_rect = text_surface.get_rect(center=rect.center)
        self.screen.blit(text_surface, text_rect)
    

    
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                    return False
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    mouse_pos = pygame.mouse.get_pos()

                    # If a prompt is visible, prioritize prompt buttons
                    if self.prompt and self.prompt_buttons:
                        accept_btn = self.prompt_buttons.get('accept')
                        decline_btn = self.prompt_buttons.get('decline')
                        if accept_btn and accept_btn.collidepoint(mouse_pos):
                            self.prompt_response = 'accept'
                            return True
                        if decline_btn and decline_btn.collidepoint(mouse_pos):
                            self.prompt_response = 'decline'
                            return True
                    
                    # Check mode buttons
                    for mode_button in self.mode_buttons:
                        if mode_button['rect'].collidepoint(mouse_pos):
                            self.mode = mode_button['mode']
                            self.current_action = -1  # Reset action when switching modes
                            self.pressed_buttons.clear()
                            break
                    
                    # Check action buttons
                    current_buttons = self.get_current_buttons()
                    for action_id, button_info in current_buttons.items():
                        if button_info['rect'].collidepoint(mouse_pos):
                            self.pressed_buttons.add(action_id)
                            break
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Left mouse button
                    # Clear all pressed buttons when mouse is released
                    self.pressed_buttons.clear()
        
        # Update current action based on pressed buttons
        if self.pressed_buttons:
            # Use the first pressed button if multiple are pressed
            self.current_action = next(iter(self.pressed_buttons))
        else:
            self.current_action = -1
        
        return True
    
    def draw(self):
        """Draw the GUI"""
        # Clear screen
        self.screen.fill(self.colors['bg'])
        
        # Draw title
        title_text = self.font_medium.render("Robotic Arm Controller", True, self.colors['text_dark'])
        title_rect = title_text.get_rect(center=(self.width // 2, 15))
        self.screen.blit(title_text, title_rect)
        
        # Draw mode buttons
        for i, mode_button in enumerate(self.mode_buttons):
            is_active = (i == self.mode)
            self.draw_mode_button(mode_button, is_active)
        
        # Draw action buttons for current mode
        current_buttons = self.get_current_buttons()
        for action_id, button_info in current_buttons.items():
            is_pressed = action_id in self.pressed_buttons
            self.draw_button(button_info, action_id, is_pressed)
        
        # Draw current action info
        if self.current_action != -1:
            action_text = f"Action: {self.action_dict.get(self.current_action, 'Unknown')}"
            action_surface = self.font_small.render(action_text, True, self.colors['text_dark'])
            self.screen.blit(action_surface, (10, 320))
        
        # Update display
        # Draw assistant panel last (always visible)
        self._draw_prompt_overlay()
        pygame.display.flip()
    
    def get_robot_state(self):
        """Get current robot control state (compatible with gesture controller interface)"""
        return self.current_action, self.mode
    
    def update(self):
        """Update the GUI (call this in the main loop)"""
        if not self.handle_events():
            return False
        
        self.draw()
        self.clock.tick(60)  # 60 FPS
        return True
    
    def cleanup(self):
        """Cleanup pygame resources"""
        pygame.quit() 

    # ---------------- Prompt UI API ----------------
    def set_prompt(self, prompt_dict):
        """Set or update the current prompt to display."""
        self.prompt = dict(prompt_dict) if prompt_dict else None
        self._setup_prompt_layout()

    def clear_prompt(self):
        """Clear the current prompt and button state."""
        self.prompt = None
        self.prompt_buttons = {}
        self.prompt_response = None

    def get_prompt_response(self):
        """Return and clear the last prompt response if any."""
        resp = self.prompt_response
        self.prompt_response = None
        return resp

    def _setup_prompt_layout(self):
        """Compute button rects for the prompt overlay (always present)."""
        overlay_h = self.prompt_overlay_height
        margin = self.prompt_margin
        btn_w, btn_h = 110, 36
        overlay_rect = pygame.Rect(0, self.height - overlay_h, self.width, overlay_h)
        # Buttons positioned bottom-right area
        y = overlay_rect.bottom - margin - btn_h
        accept_x = overlay_rect.centerx - btn_w - margin
        decline_x = overlay_rect.centerx + margin
        accept_rect = pygame.Rect(accept_x, y, btn_w, btn_h)
        decline_rect = pygame.Rect(decline_x, y, btn_w, btn_h)
        self.prompt_buttons = {
            'accept': accept_rect,
            'decline': decline_rect,
            'overlay': overlay_rect,
        }

    def _wrap_text(self, text, font, max_width):
        """Word-wrap text to lines that fit within max_width."""
        if not text:
            return []
        words = text.split()
        lines = []
        cur = ""
        for w in words:
            test = w if not cur else cur + " " + w
            if font.size(test)[0] <= max_width:
                cur = test
            else:
                if cur:
                    lines.append(cur)
                # If single word too long, hard cut
                while font.size(w)[0] > max_width and len(w) > 1:
                    # Binary chop until fits
                    lo, hi = 1, len(w)
                    cut = 1
                    while lo <= hi:
                        mid = (lo + hi) // 2
                        if font.size(w[:mid])[0] <= max_width:
                            cut = mid
                            lo = mid + 1
                        else:
                            hi = mid - 1
                    lines.append(w[:cut])
                    w = w[cut:]
                cur = w
        if cur:
            lines.append(cur)
        return lines

    def _draw_prompt_overlay(self):
        """Render the assistant panel with word-wrapped text and buttons."""
        if not self.prompt_buttons:
            self._setup_prompt_layout()
        overlay_rect = self.prompt_buttons['overlay']
        # Background
        bg_col = (30, 30, 30)
        border_col = self.colors['border']
        pygame.draw.rect(self.screen, bg_col, overlay_rect)
        pygame.draw.rect(self.screen, border_col, overlay_rect, 2)

        # Title
        title_surface = self.font_medium.render("Assistant", True, (255, 255, 255))
        self.screen.blit(title_surface, (overlay_rect.left + self.prompt_margin, overlay_rect.top + 6))

        # Message text (wrapped)
        text = str(self.prompt.get('text', '')) if self.prompt else "No suggestions."
        max_text_width = overlay_rect.width - 2 * self.prompt_margin
        lines = self._wrap_text(text, self.font_small, max_text_width)
        # Compute how many lines can fit above buttons
        btn_area_top = self.prompt_buttons['accept'].top - 6
        available_h = btn_area_top - (overlay_rect.top + 28)
        line_h = self.font_small.get_height()
        max_lines = max(1, available_h // line_h)
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            # Add ellipsis to last line if truncated
            if not lines[-1].endswith("..."):
                ell = "..."
                # Ensure fits
                while self.font_small.size(lines[-1] + ell)[0] > max_text_width and len(lines[-1]) > 0:
                    lines[-1] = lines[-1][:-1]
                lines[-1] += ell
        y = overlay_rect.top + 28
        for ln in lines:
            surf = self.font_small.render(ln, True, (230, 230, 230))
            self.screen.blit(surf, (overlay_rect.left + self.prompt_margin, y))
            y += line_h

        # Buttons (only active if a prompt is present)
        mouse_pos = pygame.mouse.get_pos()
        for key, label in [('accept', 'Accept'), ('decline', 'Decline')]:
            rect = self.prompt_buttons[key]
            is_hover = rect.collidepoint(mouse_pos)
            enabled = bool(self.prompt)
            base_col = self.colors['mode_button'] if enabled else (80, 80, 80)
            color = self.colors['mode_button_active'] if (enabled and is_hover) else base_col
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, border_col, rect, 2)
            txt_col = self.colors['text'] if enabled else (180, 180, 180)
            txt = self.font_medium.render(label, True, txt_col)
            self.screen.blit(txt, txt.get_rect(center=rect.center))