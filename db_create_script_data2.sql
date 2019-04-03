-- MySQL Workbench Forward Engineering

SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;
SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='ONLY_FULL_GROUP_BY,STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION';

-- -----------------------------------------------------
-- Schema mydb
-- -----------------------------------------------------
-- -----------------------------------------------------
-- Schema image-ranker-collector-data2
-- -----------------------------------------------------

-- -----------------------------------------------------
-- Schema image-ranker-collector-data2
-- -----------------------------------------------------
CREATE SCHEMA IF NOT EXISTS `image-ranker-collector-data2` DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_czech_ci ;
USE `image-ranker-collector-data2` ;

-- -----------------------------------------------------
-- Table `image-ranker-collector-data2`.`images`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `image-ranker-collector-data2`.`images` (
  `id` INT(11) NOT NULL,
  `filename` VARCHAR(1024) NOT NULL,
  PRIMARY KEY (`id`))
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb4
COLLATE = utf8mb4_czech_ci;


-- -----------------------------------------------------
-- Table `image-ranker-collector-data2`.`keywords`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `image-ranker-collector-data2`.`keywords` (
  `wordnet_id` INT(11) NOT NULL,
  `vector_index` INT(11) NULL DEFAULT NULL,
  `description` VARCHAR(1024) NULL DEFAULT NULL,
  PRIMARY KEY (`wordnet_id`),
  UNIQUE INDEX `vector_index_UNIQUE` (`vector_index` ASC))
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb4
COLLATE = utf8mb4_czech_ci;


-- -----------------------------------------------------
-- Table `image-ranker-collector-data2`.`words`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `image-ranker-collector-data2`.`words` (
  `id` INT(11) NOT NULL AUTO_INCREMENT,
  `word` VARCHAR(256) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE INDEX `word_UNIQUE` (`word` ASC))
ENGINE = InnoDB
AUTO_INCREMENT = 93836
DEFAULT CHARACTER SET = utf8mb4
COLLATE = utf8mb4_czech_ci;


-- -----------------------------------------------------
-- Table `image-ranker-collector-data2`.`keyword_word`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `image-ranker-collector-data2`.`keyword_word` (
  `keyword_id` INT(11) NOT NULL,
  `word_id` INT(11) NOT NULL,
  PRIMARY KEY (`keyword_id`, `word_id`),
  INDEX `fk_word_id_to_words_id_idx` (`word_id` ASC),
  CONSTRAINT `fk_keyword_word_keyword_id_to_keyword_wordnet_id`
    FOREIGN KEY (`keyword_id`)
    REFERENCES `image-ranker-collector-data2`.`keywords` (`wordnet_id`)
    ON DELETE CASCADE
    ON UPDATE CASCADE,
  CONSTRAINT `fk_keyword_word_word_id_to_words_id`
    FOREIGN KEY (`word_id`)
    REFERENCES `image-ranker-collector-data2`.`words` (`id`)
    ON DELETE CASCADE
    ON UPDATE CASCADE)
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb4
COLLATE = utf8mb4_czech_ci;


-- -----------------------------------------------------
-- Table `image-ranker-collector-data2`.`keywords_hypernyms`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `image-ranker-collector-data2`.`keywords_hypernyms` (
  `keyword_id` INT(11) NOT NULL,
  `hypernym_id` INT(11) NOT NULL,
  PRIMARY KEY (`keyword_id`, `hypernym_id`),
  INDEX `fk_keywords_hypernyms_hypernym_id_to_keywords_wordnet_id_idx` (`hypernym_id` ASC),
  CONSTRAINT `fk_keywords_hypernyms_hypernym_id_to_keywords_wordnet_id`
    FOREIGN KEY (`hypernym_id`)
    REFERENCES `image-ranker-collector-data2`.`keywords` (`wordnet_id`)
    ON DELETE CASCADE
    ON UPDATE CASCADE,
  CONSTRAINT `fk_keywords_hypernyms_keyword_id_to_keywords_wordnet_id`
    FOREIGN KEY (`keyword_id`)
    REFERENCES `image-ranker-collector-data2`.`keywords` (`wordnet_id`)
    ON DELETE CASCADE
    ON UPDATE CASCADE)
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb4
COLLATE = utf8mb4_czech_ci;


-- -----------------------------------------------------
-- Table `image-ranker-collector-data2`.`keywords_hyponyms`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `image-ranker-collector-data2`.`keywords_hyponyms` (
  `keyword_id` INT(11) NOT NULL,
  `hyponym_id` INT(11) NOT NULL,
  PRIMARY KEY (`keyword_id`, `hyponym_id`),
  INDEX `fk_keywords_hyponyms_hyponym_id_to_keywords_wordnet_id_idx` (`hyponym_id` ASC),
  CONSTRAINT `fk_keywords_hyponyms_hyponym_id_to_keywords_wordnet_id`
    FOREIGN KEY (`hyponym_id`)
    REFERENCES `image-ranker-collector-data2`.`keywords` (`wordnet_id`)
    ON DELETE CASCADE
    ON UPDATE CASCADE,
  CONSTRAINT `fk_keywords_hyponyms_keyword_id_to_keywords_wordnet_id`
    FOREIGN KEY (`keyword_id`)
    REFERENCES `image-ranker-collector-data2`.`keywords` (`wordnet_id`)
    ON DELETE CASCADE
    ON UPDATE CASCADE)
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb4
COLLATE = utf8mb4_czech_ci;


-- -----------------------------------------------------
-- Table `image-ranker-collector-data2`.`probability_vectors`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `image-ranker-collector-data2`.`probability_vectors` (
  `image_id` INT(11) NOT NULL,
  `vector_index` INT(11) NOT NULL,
  `probability` FLOAT NULL DEFAULT NULL,
  PRIMARY KEY (`image_id`, `vector_index`),
  INDEX `fk_probability_vectors_vector_index_to_keyword_vector_index_idx` (`vector_index` ASC),
  CONSTRAINT `fk_probability_vectors_image_id_to_images_id`
    FOREIGN KEY (`image_id`)
    REFERENCES `image-ranker-collector-data2`.`images` (`id`)
    ON DELETE CASCADE
    ON UPDATE CASCADE,
  CONSTRAINT `fk_probability_vectors_vector_index_to_keyword_vector_index`
    FOREIGN KEY (`vector_index`)
    REFERENCES `image-ranker-collector-data2`.`keywords` (`vector_index`)
    ON DELETE CASCADE
    ON UPDATE CASCADE)
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb4
COLLATE = utf8mb4_czech_ci;


-- -----------------------------------------------------
-- Table `image-ranker-collector-data2`.`queries`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `image-ranker-collector-data2`.`queries` (
  `id` INT(11) NOT NULL AUTO_INCREMENT,
  `query` LONGTEXT CHARACTER SET 'utf8mb4' COLLATE 'utf8mb4_bin' NOT NULL,
  `image_id` INT(11) NOT NULL,
  `type` INT(11) NOT NULL DEFAULT 0,
  `created` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP(),
  PRIMARY KEY (`id`),
  INDEX `fkImageId_idx` (`image_id` ASC),
  CONSTRAINT `fk_queries_image_id_to_images_id`
    FOREIGN KEY (`image_id`)
    REFERENCES `image-ranker-collector-data2`.`images` (`id`)
    ON UPDATE CASCADE)
ENGINE = InnoDB
AUTO_INCREMENT = 126
DEFAULT CHARACTER SET = utf8mb4
COLLATE = utf8mb4_czech_ci;


SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;
