-- MySQL Workbench Forward Engineering

SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;
SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='ONLY_FULL_GROUP_BY,STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION';

-- -----------------------------------------------------
-- Schema mydb
-- -----------------------------------------------------
-- -----------------------------------------------------
-- Schema image-ranker-collector-data
-- -----------------------------------------------------

-- -----------------------------------------------------
-- Schema image-ranker-collector-data
-- -----------------------------------------------------
CREATE SCHEMA IF NOT EXISTS `image-ranker-collector-data` DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_czech_ci ;
USE `image-ranker-collector-data` ;

-- -----------------------------------------------------
-- Table `image-ranker-collector-data`.`images`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `image-ranker-collector-data`.`images` (
  `id` INT(11) NOT NULL,
  `filename` VARCHAR(1024) NOT NULL,
  PRIMARY KEY (`id`))
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb4
COLLATE = utf8mb4_czech_ci;


-- -----------------------------------------------------
-- Table `image-ranker-collector-data`.`keywords`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `image-ranker-collector-data`.`keywords` (
  `wordnet_id` INT(11) NOT NULL,
  `vector_index` INT(11) NULL DEFAULT NULL,
  `description` VARCHAR(1024) NULL DEFAULT NULL,
  PRIMARY KEY (`wordnet_id`),
  UNIQUE INDEX `vector_index_UNIQUE` (`vector_index` ASC) VISIBLE)
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb4
COLLATE = utf8mb4_czech_ci;


-- -----------------------------------------------------
-- Table `image-ranker-collector-data`.`words`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `image-ranker-collector-data`.`words` (
  `id` INT(11) NOT NULL AUTO_INCREMENT,
  `word` VARCHAR(256) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE INDEX `word_UNIQUE` (`word` ASC) VISIBLE)
ENGINE = InnoDB
AUTO_INCREMENT = 90227
DEFAULT CHARACTER SET = utf8mb4
COLLATE = utf8mb4_czech_ci;


-- -----------------------------------------------------
-- Table `image-ranker-collector-data`.`keyword_word`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `image-ranker-collector-data`.`keyword_word` (
  `keyword_id` INT(11) NOT NULL,
  `word_id` INT(11) NOT NULL,
  PRIMARY KEY (`keyword_id`, `word_id`),
  INDEX `fk_word_id_to_words_id_idx` (`word_id` ASC) VISIBLE,
  CONSTRAINT `fk_keyword_word_keyword_id_to_keyword_wordnet_id`
    FOREIGN KEY (`keyword_id`)
    REFERENCES `image-ranker-collector-data`.`keywords` (`wordnet_id`)
    ON DELETE CASCADE
    ON UPDATE CASCADE,
  CONSTRAINT `fk_keyword_word_word_id_to_words_id`
    FOREIGN KEY (`word_id`)
    REFERENCES `image-ranker-collector-data`.`words` (`id`)
    ON DELETE CASCADE
    ON UPDATE CASCADE)
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb4
COLLATE = utf8mb4_czech_ci;


-- -----------------------------------------------------
-- Table `image-ranker-collector-data`.`keywords_hypernyms`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `image-ranker-collector-data`.`keywords_hypernyms` (
  `keyword_id` INT(11) NOT NULL,
  `hypernym_id` INT(11) NOT NULL,
  PRIMARY KEY (`keyword_id`, `hypernym_id`),
  INDEX `fk_keywords_hypernyms_hypernym_id_to_keywords_wordnet_id_idx` (`hypernym_id` ASC) VISIBLE,
  CONSTRAINT `fk_keywords_hypernyms_hypernym_id_to_keywords_wordnet_id`
    FOREIGN KEY (`hypernym_id`)
    REFERENCES `image-ranker-collector-data`.`keywords` (`wordnet_id`)
    ON DELETE CASCADE
    ON UPDATE CASCADE,
  CONSTRAINT `fk_keywords_hypernyms_keyword_id_to_keywords_wordnet_id`
    FOREIGN KEY (`keyword_id`)
    REFERENCES `image-ranker-collector-data`.`keywords` (`wordnet_id`)
    ON DELETE CASCADE
    ON UPDATE CASCADE)
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb4
COLLATE = utf8mb4_czech_ci;


-- -----------------------------------------------------
-- Table `image-ranker-collector-data`.`keywords_hyponyms`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `image-ranker-collector-data`.`keywords_hyponyms` (
  `keyword_id` INT(11) NOT NULL,
  `hyponym_id` INT(11) NOT NULL,
  PRIMARY KEY (`keyword_id`, `hyponym_id`),
  INDEX `fk_keywords_hyponyms_hyponym_id_to_keywords_wordnet_id_idx` (`hyponym_id` ASC) VISIBLE,
  CONSTRAINT `fk_keywords_hyponyms_hyponym_id_to_keywords_wordnet_id`
    FOREIGN KEY (`hyponym_id`)
    REFERENCES `image-ranker-collector-data`.`keywords` (`wordnet_id`)
    ON DELETE CASCADE
    ON UPDATE CASCADE,
  CONSTRAINT `fk_keywords_hyponyms_keyword_id_to_keywords_wordnet_id`
    FOREIGN KEY (`keyword_id`)
    REFERENCES `image-ranker-collector-data`.`keywords` (`wordnet_id`)
    ON DELETE CASCADE
    ON UPDATE CASCADE)
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb4
COLLATE = utf8mb4_czech_ci;


-- -----------------------------------------------------
-- Table `image-ranker-collector-data`.`probability_vectors`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `image-ranker-collector-data`.`probability_vectors` (
  `image_id` INT(11) NOT NULL,
  `vector_index` INT(11) NOT NULL,
  `probability` FLOAT NULL DEFAULT NULL,
  PRIMARY KEY (`image_id`, `vector_index`),
  INDEX `fk_probability_vectors_vector_index_to_keyword_vector_index_idx` (`vector_index` ASC) VISIBLE,
  CONSTRAINT `fk_probability_vectors_image_id_to_images_id`
    FOREIGN KEY (`image_id`)
    REFERENCES `image-ranker-collector-data`.`images` (`id`)
    ON DELETE CASCADE
    ON UPDATE CASCADE,
  CONSTRAINT `fk_probability_vectors_vector_index_to_keyword_vector_index`
    FOREIGN KEY (`vector_index`)
    REFERENCES `image-ranker-collector-data`.`keywords` (`vector_index`)
    ON DELETE CASCADE
    ON UPDATE CASCADE)
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb4
COLLATE = utf8mb4_czech_ci;


-- -----------------------------------------------------
-- Table `image-ranker-collector-data`.`queries`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `image-ranker-collector-data`.`queries` (
  `id` INT(11) NOT NULL AUTO_INCREMENT,
  `query` LONGTEXT CHARACTER SET 'utf8mb4' COLLATE 'utf8mb4_bin' NOT NULL,
  `image_id` INT(11) NOT NULL,
  `type` INT(11) NOT NULL DEFAULT 0,
  `created` DATETIME GENERATED ALWAYS AS (current_timestamp()) VIRTUAL,
  PRIMARY KEY (`id`),
  INDEX `fkImageId_idx` (`image_id` ASC) VISIBLE,
  CONSTRAINT `fk_queries_image_id_to_images_id`
    FOREIGN KEY (`image_id`)
    REFERENCES `image-ranker-collector-data`.`images` (`id`)
    ON UPDATE CASCADE)
ENGINE = InnoDB
AUTO_INCREMENT = 15
DEFAULT CHARACTER SET = utf8mb4
COLLATE = utf8mb4_czech_ci;


SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;