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

    externalLinksInNewWindow(); // Open external links in new window
    resizeVideo(); // Resize ".video" class on different screens

    /*** Open external links in new window ***/
    function externalLinksInNewWindow(){
      var postLinks = document.querySelectorAll('.page-content a');
      var linksLen = postLinks.length;

      for(var i = 0; i < linksLen; i++){
        if(postLinks[i].getAttribute('href').indexOf('http') != -1) {
          postLinks[i].setAttribute('target', '_blank');
        }
      };
    }

    /*** Resize ".video" class on different screens ***/
    function resizeVideo(){
      var videoElements = document.querySelectorAll('.page-content .video > iframe');

      for(var i = 0; i < videoElements.length; i++){
        var width = videoElements[i].offsetWidth; // get iframe width
        var height = (width * 0.75).toPrecision(1); // calc hight with 4:3 ratio
        videoElements[i].style.height = height + 'px'; // set height
      }
    }
  }
};

document.addEventListener("DOMContentLoaded", pageController.init());