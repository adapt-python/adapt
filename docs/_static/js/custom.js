


var docs = document.getElementsByClassName("field-list simple");
var tables = document.getElementsByClassName("longtable docutils align-default");
var className = document.getElementsByClassName("py class");

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

attributes();	
//addLinks();

