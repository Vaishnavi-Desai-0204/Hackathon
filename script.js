const inventory = document.querySelector('#images');

function getImages() {
  fetch('/images')
    .then(response => response.json())
    .then(images => {
      inventory.innerHTML = '';
      images.forEach(image => {
        const img = document.createElement('div');
        img.className = 'image';
        img.innerHTML = `<img src="${image.path}" alt="${image.name}">`;
        inventory.appendChild(img);
      });
    })
    .catch(error => console.error(error));
}

getImages();
