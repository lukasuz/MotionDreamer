function initAnimation(delay = 200) {

    cols = ['blue', 'green', 'yellow', 'red'];
    counter = 0;

    elements = document.getElementsByClassName('animated');
    len = elements.length;
    for (let i = 0; i < len; i++) {
        h1 = elements[i];
        h1.innerHTML = h1.innerHTML
            .split('')
            .map(letter => {
              return '<span>' + letter + '</span>';
            })
            .join('');
    
        Array.from(h1.children).forEach((span, index) => {
            setTimeout(() => {
                span.classList.add('wavy');
                span.style.color = cols[counter];
                counter = (counter + 1) % 4;
            }, index * 60 + delay);
        });
    }
    
}

window.addEventListener("load", (event) => {
    setTimeout(() => {
        initAnimation();
    }, 200);
    
});
