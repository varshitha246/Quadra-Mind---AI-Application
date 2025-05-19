// General utility functions
document.addEventListener('DOMContentLoaded', function() {
    // Add any client-side functionality needed
    console.log('Application loaded');
    
    // File upload previews
    const fileInputs = document.querySelectorAll('input[type="file"]');
    fileInputs.forEach(input => {
        input.addEventListener('change', function(e) {
            const fileName = e.target.files[0]?.name || 'No file selected';
            const label = this.nextElementSibling || this.parentElement.querySelector('label');
            if (label) {
                label.textContent = fileName;
            }
        });
    });
    
    // Textarea auto-resize
    const textareas = document.querySelectorAll('textarea');
    textareas.forEach(textarea => {
        textarea.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = (this.scrollHeight) + 'px';
        });
        
        // Trigger initial resize
        const event = new Event('input');
        textarea.dispatchEvent(event);
    });
});