const video_paths = {} 

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

class VideoHandler {
    constructor() {
        this.objects = [
            ['orc1', 'Orc Head (NJF)', 'Jaka Ardian 3D art (Model from Indonesia)'],
            ['orc2', 'Orc Head (NJF)', 'Jaka Ardian 3D art (Model from Indonesia)'],
            ['truck1', 'Truck (NJF)', 'Mildenhall et al. 2020'], 
            ['truck2', 'Truck (NJF)', 'Mildenhall et al. 2020'], 
            ['flame1', 'Human Face (FLAME)', 'Li et al. 2017'],
            ['flame2', 'Human Face (FLAME)', 'Li et al. 2017'],
            ['bunny', 'Bunny (NJF)', 'The Stanford 3D Scanning Repository'],
            ['smpl1', 'Human (SMPL)', 'Loper et al. 2015'],
            ['smpl2', 'Human (SMPL)', 'Loper et al. 2015'],
            ['raptor', 'Raptor (NJF)', 'TurboSquid'],
            ['owl', 'Owl (NJF)', 'TurboSquid'],
            ['smal1', 'Horse (SMAL)', 'Zuffi et al. 2017'],
            ['smal2', 'Horse (SMAL)', 'Zuffi et al. 2017'],
            
        ]
        this.id = 'video'
        this.current_animation = 0;
        this.current_view = 0;
        this.textured = false;

        this.source_field = document.getElementById('source');

        this.text_field = document.getElementById('title_video');

        this.section = document.getElementById(this.id);
        this.video = this.section.querySelector('video');
        this.source = this.section.querySelector('source');
        this.button_R = this.section.querySelector('#button-R');
        this.button_R.addEventListener('click', (event) => {
            this.current_animation = (this.current_animation + 1) % this.objects.length;
            this.change_video()

        });

        this.button_L = this.section.querySelector('#button-L');
        this.button_L.addEventListener('click', (event) => {
            this.current_animation = (this.current_animation - 1);
            if (this.current_animation < 0) {
                this.current_animation = this.objects.length - 1;
            }
            this.change_video()
        });

        this.button_T = this.section.querySelector('#button-T');
        this.button_T.addEventListener('click', (event) => {
            this.textured = !this.textured;
            this.current_view = 0;
            this.change_video()
        });

        this.button_V = this.section.querySelector('#button-V');
        this.button_V.addEventListener('click', (event) => {
            this.current_view = (this.current_view + 1) % 2;
            this.textured = false;
            this.change_video()
        });

    }

    change_video() {
        let path = this.get_path();
        
        // Fix video size while loading
        let height = this.video.clientHeight;
        let width = this.video.clientWidth;

        this.video.setAttribute('height', height);
        this.video.setAttribute('width', width);

        this.source.setAttribute('src', path);
        this.video.load();
        this.video.play();
        
        // Once loaded, remove fixed video size
        this.video.onloadeddata = () => {
            this.video.removeAttribute('height');
            this.video.removeAttribute('width');
        }

        let title = this.objects[this.current_animation][1];
        this.text_field.innerHTML = title;

        let source = this.objects[this.current_animation][2];
        this.source_field.innerHTML = source;
        
    }

    get_path() {
        let s= "";
        if (this.textured) {
            s = "tex";
        } else {
            s = "geo" + (this.current_view + 1);
        }
        let path = "./static/videos/" + this.objects[this.current_animation][0] + "/" + s + ".mp4";
        console.log(path);
        return path;
    }
}

window.addEventListener("load", (event) => {
    let vd = new VideoHandler();
    setTimeout(() => {
        initAnimation();
        vd.video.play();
    }, 200);
    
});
