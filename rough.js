// // Get references to the conveyer container and all the items
// var container = document.querySelector('.conveyer-container');
// var items = document.querySelectorAll('.outfit');

// // Add event listeners to the container for scroll events
// container.addEventListener('scroll', function() {
//     // Loop through all the items and calculate their distance from the center of the container
//     for (var i = 0; i < items.length; i++) {
//         var item = items[i];
//         var itemRect = item.getBoundingClientRect();
//         var containerRect = container.getBoundingClientRect();
//         var center = containerRect.left + containerRect.width / 2;
//         var distance = Math.abs(itemRect.left + itemRect.width / 2 - center);

//         // If the item is within a certain distance from the center, add the 'selected' class to it
//         if (distance < itemRect.width / 2) {
//             item.classList.add('selected');
//         } else {
//             item.classList.remove('selected');
//         }
//     }
// });

// CODE THAT DYNAMICALLY ADDS OUTFITS IN THE CONVEYOR-BELT CONTAINER
// Array of outfit image paths
const numOutfits = 7;
const outfitImages = [
	"clothing1.png",
	"clothing2.png",
	"blouse.png",
    "blouse.png",
    "blouse.png",
    "blouse.png",
    "blouse.png"
	// Add more image paths here as needed
];


const outfitsContainer = document.getElementById("outfits-container");

// Loop through the number of outfits and add an outfit element for each one
for (let i = 1; i < numOutfits; i++) {
	// Create a new outfit element
	const outfitElement = document.createElement("div");
	outfitElement.classList.add("outfit");

	// Create the rod, hanger, and top elements
	const rodElement = document.createElement("img");
	rodElement.classList.add("rod");
	rodElement.src = "rodfinal.png";
	outfitElement.appendChild(rodElement);

	const hangerElement = document.createElement("img");
	hangerElement.classList.add("hanger");
	hangerElement.src = "hanger1.png";
	outfitElement.appendChild(hangerElement);

	const topElement = document.createElement("img");
	topElement.classList.add("top");
	topElement.src = outfitImages[i % outfitImages.length];
	outfitElement.appendChild(topElement);

	// Add the outfit element to the container
	outfitsContainer.appendChild(outfitElement);
}



// CODE THAT CONTROLS HOVER ROTATION OF OUTFITS
const outfits = document.querySelectorAll('.outfit');

outfits.forEach(outfit => {
  outfit.addEventListener('mouseover', () => {
    outfit.querySelector('.hanger').style.transform = 'rotate3d(0, 0, 0, 0deg)';
    outfit.querySelector('.top').style.transform = 'rotate3d(0, 0, 0, 0deg)';
  });

  outfit.addEventListener('mouseout', () => {
    outfit.querySelector('.hanger').style.transform = 'rotate3d(0, 1, 0, 80deg)';
    outfit.querySelector('.top').style.transform = 'rotate3d(0, 1, 0, 80deg)';
  });
});


