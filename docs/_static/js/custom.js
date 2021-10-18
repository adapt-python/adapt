var docs = document.getElementsByClassName("field-list simple");
var code = document.getElementsByClassName("highlight");
var links = document.getElementsByClassName("wy-menu wy-menu-vertical");



const ponctList = [",", ".", ";", ":"]
//var tables = document.getElementsByClassName("longtable docutils align-default");
//var className = document.getElementsByClassName("py class");

function attributes() {
	
	if (docs.length == 0) {
        docs = document.getElementsByClassName("field-list simple");
		setTimeout(attributes, 50);
        return;
    } else {
		docs[0].appendChild(docs[1].children[0]);
		docs[0].appendChild(docs[1].children[0]);
	};
};


function changeLinks() {
	
	if (links.length == 0) {
        links = document.getElementsByClassName("wy-menu wy-menu-vertical");
		setTimeout(changeLinks, 50);
        return;
    } else {
		links[0].innerHTML = links[0].innerHTML.replace(/##/g, "#")
	};
};

function addLinks() {
	
	if (tables.length == 0 || className.length ==0) {
        tables = document.getElementsByClassName("longtable docutils align-default");
		className = document.getElementsByClassName("py class");
		setTimeout(addLinks, 50);
        return;
    } else {
		var tbody = tables[0].children[1];
		for (var i = 0; i < tbody.children.length; i++) {
			var splits = tbody.children[i].innerHTML.split("<span class=\"pre\">");
			splits[1] = splits[1].split("</span>");
			console.log(splits);
			newInner = splits[0].concat("<span class=\"pre\"><a href=#").concat(className[0].children[0].id).concat(".").concat(splits[1][0]).concat(">").concat(splits[1][0]).concat("</a></span>").concat(splits[1][1]);
			tbody.children[i].innerHTML = newInner;
			console.log(newInner);
		};
		window.scrollTo(0,0);
	};
};

function changeColor() {
  
  if (code.length == 0) {
        code = document.getElementsByClassName("highlight");
		setTimeout(changeColor, 50);
        return;
    } 
  else {
	  for (var k = 0; k < code.length; k++) {
		  var elements = code[k].children[0].children;
		  var isPoint = false
		  for (var i = 0; i < elements.length; i++) {
			  var elem = elements[i]
			  
			  if (isPoint == true) {
				elem.style.color = "#0e84b5";
			  }
			  
			  isPoint = false
			  
			  if (elem.className == "o" && ponctList.includes(elem.innerHTML)) {
				elem.style.color = "#212529";
				elem.style.fontWeight = "normal";
				
				if (elem.innerHTML == ".") {
					isPoint = true;
				}
				
				}
			  }
		  }
  }
}

changeColor();
attributes();
//changeLinks();
//addLinks();

