pipeline {
  agent any
  stages {
    stage('build') {
      steps {
        git(url: 'https://github.com/fastestimator/fastestimator', branch: 'master')
      }
    }
  }
}