angular.module('app', ['ngRoute'])

  .controller('TodoController', ['$scope', function ($scope) {
    $scope.todos = [
      { name: 'AngularJS Directives', completed: true },
      { name: 'Data binding', completed: true },
      { name: '$scope', completed: true },
      { name: 'Controllers and Modules', completed: true },
      { name: 'Templates and routes', completed: true },
      { name: 'Filters and Services', completed: false },
      { name: 'Get started with Node/ExpressJS', completed: false },
      { name: 'Setup MongoDB database', completed: false },
      { name: 'Be awesome!', completed: false },
    ];
  }])
  
  .config(['$routeProvider', function ($routeProvider) {
    console.log('shouldnt be here');
    $routeProvider
      .when('/', {
        templateUrl: 'todos_routes.html',
        controller: 'TodoController'
      })
  }]);
