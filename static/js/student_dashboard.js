/**
 * Student Dashboard Visuals
 */
document.addEventListener('DOMContentLoaded', () => {
    // Render progress circle indicators
    const progressRings = document.querySelectorAll('.progress-ring-circle');
    
    progressRings.forEach(ring => {
        const radius = ring.r.baseVal.value;
        const circumference = radius * 2 * Math.PI;
        
        // Retrieve target percent
        const percent = parseFloat(ring.getAttribute('data-percent'));
        
        // Stroke offset
        ring.style.strokeDasharray = `${circumference} ${circumference}`;
        ring.style.strokeDashoffset = circumference;
        
        const offset = circumference - (percent / 100 * circumference);
        
        // Animate offset transition
        setTimeout(() => {
            ring.style.strokeDashoffset = offset;
        }, 300);
    });
});
