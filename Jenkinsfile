pipeline {
  agent any
  
  stages {
    stage('Build') {
      steps {
        sh './test/install_dependencies.sh'
      }
    }
    stage('Test') {
        steps {
            sh 'python -m pytest'
        }
    }

  }
}
