/* app.js - All JavaScript functionality for the Streamlit application */

// Mobile detection and responsive behavior
function initializeMobileDetection() {
    window.addEventListener('resize', function() {
        const width = window.innerWidth;
        if (width < 640) {
            // Mobile view - could add custom mobile behavior here
            console.log('Mobile view detected');
        }
    });
}

// Auto-scroll chat to bottom
function autoScrollChat() {
    const chatBox = window.parent.document.getElementById('chat-box');
    if (chatBox) { 
        chatBox.scrollTop = chatBox.scrollHeight; 
    }
}

// Enhanced textarea auto-resize functionality
function initializeTextareaResize() {
    // Find the textarea and apply enhanced auto-resize functionality
    const ta = window.parent.document.querySelector('textarea');
    if (!ta) return;
    
    // Set initial properties
    ta.style.overflow = 'hidden';
    ta.style.minHeight = '55px'; // Match our CSS
    ta.style.maxHeight = '120px'; // Match our CSS
    
    const resize = () => {
        // Reset height to measure scroll height
        ta.style.height = 'auto';
        
        // Calculate new height with constraints
        const newHeight = Math.max(55, Math.min(ta.scrollHeight, 120));
        ta.style.height = newHeight + 'px';
        
        // Show scrollbar if content exceeds max height
        if (ta.scrollHeight > 120) {
            ta.style.overflowY = 'auto';
        } else {
            ta.style.overflowY = 'hidden';
        }
    };
    
    // Attach event listeners
    ta.addEventListener('input', resize);
    ta.addEventListener('paste', () => setTimeout(resize, 10));
    ta.addEventListener('focus', resize);
    
    // Initial resize
    setTimeout(resize, 100);
    
    // Resize on window resize to maintain responsiveness
    window.addEventListener('resize', resize);
}

// Initialize all JavaScript functionality
function initializeApp() {
    initializeMobileDetection();
    initializeTextareaResize();
}

// Auto-initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeApp);
} else {
    initializeApp();
}