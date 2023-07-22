function showLoadingIndicator() {
	document.getElementById("loading-indicator").style.display = "flex";
}

function auto_grow(element) {
	element.style.height = "0px";
	element.style.height = element.scrollHeight + "px";
}
