'use strict';
var navigation = new Navigation();
var modal = new Modal();

/** MAIN PAGE **/
var pageController = {

  /** Init site functions **/
  init: function (){
    navigation.init(); // initialization of main (top) menu
    modal.init(); // initialization of modal form controller
    this.pageHandlers(); // init different page functions
  },

  /** Page functions **/
  pageHandlers: function (){
    var postLinks = document.querySelectorAll('.page-content a');
    var linksLen = postLinks.length;

    for(var i = 0; i < linksLen; i++){
      if(postLinks[i].getAttribute('href').indexOf('http') != -1) {
        postLinks[i].setAttribute('target', '_blank');
      }
    };
  }
};

document.addEventListener("DOMContentLoaded", pageController.init());