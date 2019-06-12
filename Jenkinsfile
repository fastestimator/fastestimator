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
            sh '. venv/bin/activate'
            sh 'python3 -m pytest'
        }
    }

  }
}
