CREATE TABLE `predictions` (
	`id` int AUTO_INCREMENT NOT NULL,
	`userId` int NOT NULL,
	`employeeLabel` varchar(120) NOT NULL,
	`mode` varchar(32) NOT NULL,
	`threshold` varchar(8) NOT NULL,
	`probability` varchar(16) NOT NULL,
	`predictedAttrition` int NOT NULL,
	`inputs` text NOT NULL,
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	CONSTRAINT `predictions_id` PRIMARY KEY(`id`)
);
