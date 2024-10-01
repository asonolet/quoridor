# TODO

## Game env

- [x] fix dok_matrix related code
- [x] profile and check if it is well optimized
- [x] refactor methods taking boolean arguments
- [x] setting up small documentation
- [o] using dev branch and jenkins integration tests in master
    - [x] use pytest pre-commit to check everything is working
    - [ ] add full test on game
- [x] fixing most of ruff warnings
- [x] debug wall number
- [ ] wrap game and bs in RL terms (env, action, actor, policy, ...)

## RL

- [ ] learn how to score greedily
      - make database of full play
      - use NN to learn state score each time it was computed
